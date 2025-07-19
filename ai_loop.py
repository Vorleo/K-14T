#!/usr/bin/env python3
"""
K-14T Fast Loop (Full persona once, micro persona thereafter)

Features:
- Loads full persona from persona/system_prompt.txt ONCE (first user turn).
- After first turn uses a minimal in-code MICRO_PERSONA instead of repeating full text.
- Optional interval-based reminder injection to keep the model on-style.
- Limited history (last N user turns; assistant mirrored).
- Memory relevance selection (top 3 simple overlap).
- Output cap via num_predict for faster replies.
- Timing instrumentation.
- Fast mode toggle (/fast) lowers num_predict & disables voice.
"""

import os
import sys
import time
import json
import pathlib
import argparse
import requests
from collections import deque

# -------------------- Config / Paths --------------------
BASE_DIR = pathlib.Path(__file__).parent
CONFIG_PATH = BASE_DIR / "config.json"
PERSONA_PATH = BASE_DIR / "persona" / "system_prompt.txt"
MEMORY_FILE = BASE_DIR / "memory" / "long_term.jsonl"

DEFAULT_CONFIG = {
    "model": "phi:latest",
    "history_turns": 2,
    "callsign": "Joshua",
    "voice": False,
    "beeps": False,
    "num_predict": 60,
    "temperature": 0.6,
    "top_p": 0.9
}

# ------------- Micro Persona & Reminder Strategy ----------
MICRO_PERSONA = (
    "K-14T stays in-universe, concise (≤2 short sentences), no meta or AI talk."
)
# Inject micro persona every turn after first? (Set False to use interval logic)
INJECT_MICRO_PERSONA_ALWAYS = True
# If ALWAYS is False, inject on these modular intervals (e.g., every 4 turns)
REMINDER_INTERVAL = 4

# -------------------- Load Config ------------------------
def load_config():
    if CONFIG_PATH.exists():
        try:
            data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            return {**DEFAULT_CONFIG, **data}
        except Exception:
            print("[WARN] Bad config.json; defaults used.")
    return dict(DEFAULT_CONFIG)

def save_config(cfg):
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

# -------------------- Persona Handling -------------------
_persona_warned = False
def load_full_persona():
    global _persona_warned
    if PERSONA_PATH.exists():
        txt = PERSONA_PATH.read_text(encoding="utf-8").strip()
        return txt
    if not _persona_warned:
        print("[WARN] persona/system_prompt.txt missing; only micro persona will be used.")
        _persona_warned = True
    return ""

# -------------------- Memory -----------------------------
_memory_cache = []
_memory_norm_set = set()

def _norm(s: str) -> str:
    return " ".join(s.lower().split())

def load_memory():
    if not MEMORY_FILE.exists():
        return
    for ln in MEMORY_FILE.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        try:
            obj = json.loads(ln)
            fact = obj.get("fact") or obj.get("text") or ""
        except Exception:
            fact = ln.strip()
        if fact:
            n = _norm(fact)
            if n not in _memory_norm_set:
                _memory_cache.append(fact)
                _memory_norm_set.add(n)

def append_memory(fact: str):
    fact = fact.strip()
    if not fact:
        return False
    n = _norm(fact)
    if n in _memory_norm_set:
        return False
    entry = {"ts": time.time(), "fact": fact}
    MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(MEMORY_FILE, "a", encoding="utf-8") as f:
        json.dump(entry, f); f.write("\n")
    _memory_cache.append(fact)
    _memory_norm_set.add(n)
    return True

def select_relevant(user_msg: str, limit=3):
    if not _memory_cache:
        return []
    u = set(w for w in user_msg.lower().split() if len(w) > 2)
    scored = []
    for fact in _memory_cache:
        ft = set(fact.lower().split())
        overlap = len(u & ft)
        scored.append((overlap, fact))
    scored.sort(key=lambda x: x[0], reverse=True)
    chosen = [f for o, f in scored if o > 0][:limit]
    if not chosen:
        chosen = _memory_cache[-limit:]
    return chosen[:limit]

# -------------------- Model Calls ------------------------
API_ROOT = "http://localhost:11434"
API_GENERATE = f"{API_ROOT}/api/generate"
API_TAGS = f"{API_ROOT}/api/tags"

def wait_for_server(timeout=45):
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(API_TAGS, timeout=2)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False

def call_model(model, prompt, cfg, fast_mode=False):
    npredict = 40 if fast_mode else int(cfg.get("num_predict", 60))
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": npredict,
            "temperature": cfg.get("temperature", 0.6),
            "top_p": cfg.get("top_p", 0.9)
        }
    }
    t0 = time.time()
    r = requests.post(API_GENERATE, json=payload)
    r.raise_for_status()
    t1 = time.time()
    resp = r.json().get("response", "").strip()
    print(f"[TIMING] infer={t1 - t0:.2f}s")
    return resp

# -------------------- Voice Placeholder ------------------
def speak(text, cfg):
    if not cfg.get("voice"):
        return
    print("[VOICE suppressed in perf mode]")

# -------------------- Prompt Build -----------------------
def build_prompt(history, mem_facts, user_msg, callsign,
                 full_persona="", use_micro=False):
    """
    Only include ONE of: full_persona (once) or micro persona (if use_micro True).
    """
    parts = []
    if full_persona:
        parts.append(full_persona)
    elif use_micro and MICRO_PERSONA:
        parts.append(MICRO_PERSONA)

    if mem_facts:
        parts.append("[MEM: " + "; ".join(mem_facts) + "]")

    if history:
        parts.append("\n".join(f"{r}: {t}" for r, t in history))

    parts.append(f"User ({callsign}): {user_msg}")
    parts.append("K-14T:")
    return "\n".join(p for p in parts if p).strip()

# -------------------- Reply Processing -------------------
def trim_sentences(reply, max_sentences=2):
    seps = reply.replace("!", ".").replace("?", ".")
    segs = [s.strip() for s in seps.split(".") if s.strip()]
    if len(segs) <= max_sentences:
        return reply.strip()
    return ". ".join(segs[:max_sentences]) + "."

def sanitize(reply):
    low = reply.lower()
    for bad in ("as an ai", "language model", "i am an ai"):
        if bad in low:
            reply = reply.replace(bad, "")
    return reply.strip()

# -------------------- Main Loop --------------------------
def main():
    if not wait_for_server():
        print("[ERROR] Ollama server not reachable.")
        sys.exit(1)

    cfg = load_config()
    full_persona = load_full_persona()
    load_memory()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", help="Override model")
    args = parser.parse_args()

    model = args.model if args.model else cfg.get("model", "phi:latest")
    callsign = cfg.get("callsign", "Joshua")

    print(f"K-14T loop online. Model: {model}")
    print("Commands: /model <m>, /callsign <name>, /remember <fact>, /mem, /fast, /reload, /exit")

    history_turns = int(cfg.get("history_turns", 2))
    history = deque(maxlen=history_turns * 2 * 2)  # user+assistant pairs
    fast_mode = False
    first_turn = True
    user_turn_counter = 0  # counts user *content* turns

    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            user = "/exit"

        if not user:
            continue
        lower = user.lower()

        # ---------- Commands ----------
        if lower in ("/exit", "exit", "/quit", "quit"):
            print(f"K-14T: Shutdown sequence, {callsign}.")
            break

        if user.startswith("/model"):
            parts = user.split(maxsplit=1)
            if len(parts) == 2:
                model = parts[1].strip()
                cfg["model"] = model
                save_config(cfg)
                print(f"[INFO] Model -> {model}")
            else:
                print("Usage: /model <name>")
            continue

        if user.startswith("/callsign"):
            parts = user.split(maxsplit=1)
            if len(parts) == 2:
                callsign = parts[1].strip()
                cfg["callsign"] = callsign
                save_config(cfg)
                print(f"[INFO] Callsign -> {callsign}")
            else:
                print("Usage: /callsign <name>")
            continue

        if user.startswith("/remember"):
            parts = user.split(maxsplit=1)
            if len(parts) == 2:
                fact = parts[1].strip()
                stored = append_memory(fact)
                print(f"[MEM] {'Stored' if stored else 'Duplicate'}: {fact}")
            else:
                print("Usage: /remember <fact>")
            continue

        if lower == "/mem":
            if not _memory_cache:
                print("[MEM] (none)")
            else:
                print("[MEM]")
                for i, fct in enumerate(_memory_cache[-25:], 1):
                    print(f" {i}. {fct}")
            continue

        if lower == "/fast":
            fast_mode = not fast_mode
            if fast_mode:
                print("[MODE] Fast ON (num_predict=40; voice off).")
            else:
                print("[MODE] Fast OFF.")
            continue

        if lower == "/reload":
            cfg = load_config()
            callsign = cfg.get("callsign", callsign)
            full_persona = load_full_persona()
            first_turn = True  # re-inject full persona after reload
            print("[INFO] Config + persona reloaded (will inject full persona next turn).")
            continue

        if user.startswith("/"):
            print("[WARN] Unknown command.")
            continue

        # ---------- Build Prompt ----------
        t_build_start = time.time()
        mem_facts = select_relevant(user, limit=3)

        user_turn_counter += 1

        # Decide persona usage
        if first_turn:
            prompt = build_prompt(history, mem_facts, user, callsign,
                                  full_persona=full_persona, use_micro=False)
        else:
            use_micro = False
            if INJECT_MICRO_PERSONA_ALWAYS:
                use_micro = True
            else:
                if REMINDER_INTERVAL > 0 and user_turn_counter % REMINDER_INTERVAL == 0:
                    use_micro = True
            prompt = build_prompt(history, mem_facts, user, callsign,
                                  full_persona="", use_micro=use_micro)

        t_build_end = time.time()

        if first_turn:
            first_turn = False  # full persona will not be resent

        # ---------- Inference ----------
        reply_raw = call_model(model, prompt, cfg, fast_mode=fast_mode)
        reply = sanitize(reply_raw)
        reply = trim_sentences(reply, max_sentences=2)

        t_total_end = time.time()
        print(f"[TIMING] build={t_build_end - t_build_start:.2f}s total={t_total_end - t_build_start:.2f}s "
              f"prompt_tokens≈{len(prompt.split())}")

        print(f"K-14T: {reply}")
        speak(reply, cfg)

        history.append(("User", user))
        history.append(("K-14T", reply))

if __name__ == "__main__":
    main()
