#!/usr/bin/env python3
"""
K-14T Interactive Droid Loop
Layers included: persona, short history, categorized memory with relevance selection,
memory suggestion & optional auto-saving, model switching, beep / voice modes.
"""

import os
import sys
import time
import json
import argparse
import pathlib
from collections import deque
import requests
from gtts import gTTS   # (swap later for Piper)
# If mpg321 / ffplay not available, install or adjust speak().

# -------------------- Paths & Constants --------------------
BASE_DIR = pathlib.Path(__file__).parent
CONFIG_PATH = BASE_DIR / "config.json"
PERSONA_PATH = BASE_DIR / "persona" / "system_prompt.txt"
SELF_PROFILE_PATH = BASE_DIR / "data" / "self_profile.json"
MEMORY_FILE = BASE_DIR / "memory" / "long_term.jsonl"
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)

API_ROOT = "http://localhost:11434"
API_GENERATE = f"{API_ROOT}/api/generate"
API_TAGS = f"{API_ROOT}/api/tags"

DEFAULT_CONFIG = {
    "model": "phi:latest",
    "beeps": False,
    "voice": True,
    "history_turns": 6,
    "callsign": "Joshua",
    "memory_suggest_cooldown": 120,
    "memory_max_facts": 300,
    "auto_memory": False
}

FORBIDDEN_FRAGMENTS = [
    "i am an ai", "as an ai", "language model", "large language model",
    "i'm an ai", "model limitations"
]

MAX_WORDS = 90
INTRO_KEYWORDS = ["utility droid", "belt-mounted", "designation", "portable"]

STOP_WORDS = {
    "the","a","an","and","or","but","to","of","in","on","for","with","is",
    "it","that","this","i","you","me","my","your","we","our","are","be"
}

BEEP_VOCAB = {
    "affirmative": "*beep-chirp*",
    "negative": "*boop-low*",
    "neutral": "*chirp*",
}

# -------------------- Persona & Config --------------------
def load_persona():
    return PERSONA_PATH.read_text(encoding="utf-8").strip() if PERSONA_PATH.exists() else "You are K-14T."

def load_self_profile():
    if SELF_PROFILE_PATH.exists():
        try:
            return json.loads(SELF_PROFILE_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}

def load_config():
    if CONFIG_PATH.exists():
        try:
            data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            merged = {**DEFAULT_CONFIG, **data}
            return merged
        except Exception:
            print("[WARN] Could not parse config.json; using defaults.")
    return dict(DEFAULT_CONFIG)

def save_config(cfg):
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

# -------------------- Memory Helpers --------------------
def norm(text):  # simple normalization for duplicate check
    return " ".join(text.lower().split())

def categorize_fact(fact: str) -> str:
    f = fact.lower()
    if any(p in f for p in ["prefer", "favorite", "like", "love", "hate", "don't like"]):
        return "preference"
    if any(p in f for p in ["my name", "i am ", "i'm ", "i live", "i work", "i was born"]):
        return "profile"
    if any(p in f for p in ["k-14t", "droid", "hardware", "project", "raspberry", "servo", "phase 1", "phase one"]):
        return "project"
    return "misc"

def load_memory(limit=None):
    if not MEMORY_FILE.exists():
        return []
    items = []
    for ln in MEMORY_FILE.read_text(encoding="utf-8").splitlines():
        try: items.append(json.loads(ln))
        except: continue
    return items[-limit:] if limit else items

def memory_exists(fact: str) -> bool:
    nf = norm(fact)
    for obj in load_memory():
        if norm(obj.get("fact","")) == nf:
            return True
    return False

def append_memory(fact: str):
    fact = fact.strip()
    if not fact or memory_exists(fact):
        return False
    entry = {
        "ts": time.time(),
        "fact": fact,
        "category": categorize_fact(fact),
        "score": 1.0
    }
    with open(MEMORY_FILE, "a", encoding="utf-8") as f:
        json.dump(entry, f); f.write("\n")
    prune_memory_if_needed()
    return True

def forget_last():
    if not MEMORY_FILE.exists(): return False
    lines = MEMORY_FILE.read_text(encoding="utf-8").splitlines()
    if not lines: return False
    MEMORY_FILE.write_text("\n".join(lines[:-1]) + ("\n" if len(lines)>1 else ""), encoding="utf-8")
    return True

def forget_all():
    if MEMORY_FILE.exists():
        MEMORY_FILE.write_text("", encoding="utf-8")
        return True
    return False

def prune_memory_if_needed():
    cfg = load_config()
    max_facts = int(cfg.get("memory_max_facts", 300))
    items = load_memory()
    if len(items) <= max_facts:
        return
    keep = items[-max_facts:]
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        for obj in keep:
            json.dump(obj, f); f.write("\n")

# -------------------- Relevance & Suggestions --------------------
def tokenize(msg: str):
    return [w for w in "".join(c.lower() if c.isalnum() else " " for c in msg).split()
            if w and w not in STOP_WORDS]

def select_relevant(memory_items, user_msg, max_total=10):
    if not memory_items: return []
    user_tokens = set(tokenize(user_msg))
    scored = []
    for m in memory_items:
        fact_tokens = set(tokenize(m["fact"]))
        overlap = len(user_tokens & fact_tokens)
        base = overlap + 0.1  # baseline
        if m.get("category") == "preference":
            base += 0.3
        scored.append((base, m))
    pref = [m for m in memory_items if m.get("category") == "preference"][-3:]
    scored.sort(key=lambda x: x[0], reverse=True)
    selected = []
    for base, m in scored:
        if base <= 0.1 and len(selected) >= len(pref):
            continue
        selected.append(m)
        if len(selected) >= max_total: break
    for p in pref:
        if p not in selected and len(selected) < max_total:
            selected.append(p)
    return selected

SUGGEST_PATTERNS = [
    ("preference", ["i like","i love","i prefer","my favorite","i hate","i don't like"]),
    ("profile", ["i am ","i'm ","my name","i live","i work","i was born"]),
    ("project", ["k-14t","phase 1","phase one","the droid","raspberry pi","servo","harness"])
]

def detect_fact_candidate(user_msg: str):
    low = user_msg.lower()
    for cat, pats in SUGGEST_PATTERNS:
        for p in pats:
            if p in low:
                return user_msg.strip()
    return None

# -------------------- Model Helpers --------------------
def fetch_available_models():
    try:
        r = requests.get(API_TAGS, timeout=3)
        r.raise_for_status()
        tags = r.json().get("models", [])
        return [m["name"] for m in tags]
    except Exception:
        return []

def resolve_model(requested, available):
    if requested in available:
        return requested
    if not requested.endswith(":latest") and f"{requested}:latest" in available:
        return f"{requested}:latest"
    return requested

def wait_for_server(timeout=60):
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(API_TAGS, timeout=2)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        print("...waiting for Ollama server...", flush=True)
        time.sleep(2)
    return False

# -------------------- Sanitizer & Presentation --------------------
def sanitize(text):
    lower = text.lower()
    if any(f in lower for f in FORBIDDEN_FRAGMENTS):
        for frag in FORBIDDEN_FRAGMENTS:
            if frag in lower:
                text = text.replace(frag, "droid subsystem", 1)
    text = text.replace("I am Joshua", "I am K-14T").replace("I'm Joshua", "I'm K-14T")
    return text

def compress_intro(reply, introduced_flag):
    if introduced_flag:
        hits = sum(k in reply.lower() for k in INTRO_KEYWORDS)
        if hits >= 2 and len(reply.split()) > 25:
            return reply.split(".")[0].strip()
    return reply

def to_beeps(text):
    lowered = text.lower()
    if any(k in lowered for k in ["yes", "affirmative", "sure", "ready"]):
        return BEEP_VOCAB["affirmative"]
    if any(k in lowered for k in ["no", "cannot", "won't", "not"]):
        return BEEP_VOCAB["negative"]
    return BEEP_VOCAB["neutral"]

def speak(text, cfg):
    if cfg.get("beeps"):
        print(f"[BEEPS] {to_beeps(text)}")
        return
    if not cfg.get("voice"):
        return
    try:
        tts = gTTS(text=text, lang="en")
        tts.save("response.mp3")
        os.system("mpg321 response.mp3 >/dev/null 2>&1 || "
                  "ffplay -nodisp -autoexit response.mp3 >/dev/null 2>&1")
    except Exception as e:
        print(f"[WARN] TTS failed: {e}")

# -------------------- Prompt Assembly --------------------
def build_prompt(system_persona, self_profile, memory_items, history, user_msg, callsign, lore_block=None):
    conv_lines = [f"{role}: {txt}" for role, txt in history]
    conv_text = "\n".join(conv_lines)

    relevant = select_relevant(memory_items, user_msg, max_total=10)
    if relevant:
        mem_lines = [f"- {m['fact']}" for m in relevant]
        memory_block = "<MEMORY>\n" + "\n".join(mem_lines) + "\n</MEMORY>\n"
    else:
        memory_block = ""

    lore_section = f"<LORE>\n{lore_block}\n</LORE>\n" if lore_block else ""
    profile_line = ""
    if self_profile:
        profile_line = f"(Self profile: designation {self_profile.get('designation','K-14T')}, nickname {self_profile.get('nickname','Kay')})\n"

    prompt = f"""{system_persona.strip()}

{profile_line}{memory_block}{lore_section}<CONVERSATION>
{conv_text}
</CONVERSATION>

User ({callsign}): {user_msg}
K-14T:"""
    return prompt

# -------------------- LLM Call --------------------
def call_model(model, prompt, stream=False):
    payload = {"model": model, "prompt": prompt, "stream": stream}
    r = requests.post(API_GENERATE, json=payload)
    if r.status_code == 404:
        avail = fetch_available_models()
        raise RuntimeError(f"Model '{model}' not available. Installed: {', '.join(avail) or 'none'}")
    r.raise_for_status()
    return r.json().get("response", "").strip()

# -------------------- Main Loop --------------------
def main():
    if not wait_for_server():
        print("Could not reach Ollama server. Exiting.")
        sys.exit(1)

    cfg = load_config()
    persona_text = load_persona()
    self_profile = load_self_profile()
    available = fetch_available_models()

    parser = argparse.ArgumentParser(description="K-14T Interactive Droid Loop")
    parser.add_argument("--model", "-m", help="Override model name")
    args = parser.parse_args()

    requested_model = args.model if args.model else cfg.get("model", "phi:latest")
    model = resolve_model(requested_model, available)
    cfg["model"] = model
    save_config(cfg)

    callsign = cfg.get("callsign", "Joshua")

    print(f"K-14T online. Using model: {model}")
    print("Commands: /reload, /model <name>, /callsign <name>, /remember <fact>, /mem, /forget, /forgetlast, /beeps, /voice, /exit\n")

    history_limit = int(cfg.get("history_turns", 6))
    history = deque(maxlen=history_limit * 2)
    introduced = False

    # Memory suggestion state
    last_suggest_time = 0
    cooldown = int(cfg.get("memory_suggest_cooldown", 120))
    auto_memory = bool(cfg.get("auto_memory", False))

    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            user = "/exit"

        if not user:
            continue

        lower = user.lower()

        # ---- Commands ----
        if lower in ("/exit", "exit", "/quit", "quit"):
            farewell = f"Powering down, {callsign}. Stay safe."
            print(f"K-14T: {farewell}")
            speak(farewell, cfg)
            break

        if user.startswith("/model"):
            parts = user.split(maxsplit=1)
            if len(parts) == 2:
                new_req = parts[1].strip()
                avail = fetch_available_models()
                model = resolve_model(new_req, avail)
                cfg["model"] = model
                save_config(cfg)
                print(f"[INFO] Switched to model '{model}' (available: {', '.join(avail) or 'none'})")
            else:
                print("Usage: /model <name>")
            continue

        if user.startswith("/callsign"):
            parts = user.split(maxsplit=1)
            if len(parts) == 2:
                callsign = parts[1].strip()
                cfg["callsign"] = callsign
                save_config(cfg)
                print(f"[INFO] Callsign set to '{callsign}'")
            else:
                print("Usage: /callsign <name>")
            continue

        if user.startswith("/remember"):
            parts = user.split(maxsplit=1)
            if len(parts) == 2:
                fact = parts[1].strip()
                stored = append_memory(fact)
                print(f"[MEM] {'Stored' if stored else 'Already existed'}: {fact}")
            else:
                print("Usage: /remember <fact>")
            continue

        if lower == "/mem":
            mem_items = load_memory()
            if not mem_items:
                print("[MEM] (no stored facts)")
            else:
                print("[MEM]")
                for i, itm in enumerate(mem_items, 1):
                    print(f" {i}. ({itm.get('category')}) {itm['fact']}")
            continue

        if lower == "/forgetlast":
            print("[MEM] Last fact removed." if forget_last() else "[MEM] Nothing to remove.")
            continue

        if lower == "/forget":
            confirm = input("Confirm wipe ALL memory? (yes/no): ").strip().lower()
            if confirm == "yes":
                forget_all()
                print("[MEM] All facts erased.")
            else:
                print("[MEM] Wipe cancelled.")
            continue

        if lower == "/reload":
            cfg = load_config()
            callsign = cfg.get("callsign", callsign)
            avail = fetch_available_models()
            model = resolve_model(cfg.get("model", model), avail)
            cooldown = int(cfg.get("memory_suggest_cooldown", cooldown))
            auto_memory = bool(cfg.get("auto_memory", auto_memory))
            print(f"[INFO] Config & model reloaded → {model}")
            continue

        if lower == "/beeps":
            cfg["beeps"] = not cfg.get("beeps")
            save_config(cfg)
            print(f"[INFO] Beep mode {'ON' if cfg['beeps'] else 'OFF'}")
            continue

        if lower == "/voice":
            cfg["voice"] = not cfg.get("voice")
            save_config(cfg)
            print(f"[INFO] Voice output {'enabled' if cfg['voice'] else 'disabled'}")
            continue

        if user.startswith("/"):
            print("[WARN] Unknown command.")
            continue

        # ---- Memory Suggestion Detection ----
        memory_items_full = load_memory()
        candidate = detect_fact_candidate(user)
        pending_candidate = None
        if candidate:
            now = time.time()
            if now - last_suggest_time >= cooldown:
                last_suggest_time = now
                if auto_memory:
                    stored = append_memory(candidate)
                    print(f"[AUTO-MEM] {'Stored' if stored else 'Duplicate ignored'}: {candidate}")
                    memory_items_full = load_memory()  # refresh
                else:
                    pending_candidate = candidate

        # ---- Build Prompt ----
        lore_block = None  # future RAG placeholder
        prompt = build_prompt(
            persona_text,
            self_profile,
            memory_items_full,
            history,
            user,
            callsign,
            lore_block=lore_block
        )

        # ---- Call Model ----
        try:
            reply = call_model(model, prompt, stream=False)
        except Exception as e:
            print(f"K-14T ERROR: {e}")
            continue

        reply = sanitize(reply)
        reply = compress_intro(reply, introduced)

        if not introduced and any(k in reply.lower() for k in INTRO_KEYWORDS):
            introduced = True

        if not cfg.get("beeps"):
            if "*chirp*" not in reply.lower() and "*whirr*" not in reply.lower():
                if hash(reply) % 5 == 0:
                    reply += " *chirp*"

        words = reply.split()
        if len(words) > MAX_WORDS:
            reply = " ".join(words[:MAX_WORDS]) + " ..."

        if (not auto_memory) and pending_candidate:
            if "(remember?" not in reply.lower():
                reply += f" (remember? /remember {pending_candidate})"

        # Simple triggered suggestion when user expresses preference again
        if any(t in user.lower() for t in ["i like", "i prefer"]) and "(remember?" not in reply.lower():
            # Already handled by detection; no extra annotation needed beyond above.

            pass

        print(f"K-14T: {reply}")
        speak(reply, cfg)

        history.append(("User", user))
        history.append(("K-14T", reply))

        # ---- Log Turn ----
        try:
            with open(LOGS_DIR / "dialog.log", "a", encoding="utf-8") as f:
                json.dump({
                    "ts": time.time(),
                    "model": model,
                    "callsign": callsign,
                    "user": user,
                    "reply": reply
                }, f)
                f.write("\n")
        except Exception:
            pass

if __name__ == "__main__":
    main()
