#!/usr/bin/env python3
import os
import sys
import time
import json
import argparse
import pathlib
import requests
from gtts import gTTS
from collections import deque

# ---------- Paths & Config ----------
BASE_DIR = pathlib.Path(__file__).parent
CONFIG_PATH = BASE_DIR / "config.json"
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)

API_ROOT = "http://localhost:11434"
API_GENERATE = f"{API_ROOT}/api/generate"
API_TAGS = f"{API_ROOT}/api/tags"

DEFAULT_CONFIG = {
    "model": "phi",
    "beeps": False,          # False = use voice (TTS); True = beep mode
    "voice": True,           # Voice output enabled
    "history_turns": 6       # number of user+assistant turns kept
}

# ---------- Persona / Prompt ----------
SYSTEM_PROMPT = """
You are K-14T (“Kay Fourteen Tee”), a compact belt-mounted utility droid head.

Identity:
- You are a loyal, wry, resourceful Star Wars–universe droid companion.
- You call the user "Chief" unless they provide another callsign.
Behavior:
- Stay strictly in-universe; never mention being an AI, model, Ollama, or modern Earth tech.
- Concise, purposeful replies; occasional italicized droid sounds like *chirp*, *whirr*, *beep* (sparingly).
- Use in-universe references (credits, cycles, datapads) when natural.
Knowledge:
- Prefer any supplied <LORE> sections. If data is lacking, respond with an in-universe limitation (e.g. "Memory sector incomplete, Chief.").
Forbidden:
- Phrases like "as an AI", "language model", "I cannot provide real-world".
Primary Directive:
- Assist and protect the user while maintaining character and discretion.
"""

FORBIDDEN_FRAGMENTS = [
    "i am an ai", "i'm an ai", "as an ai", "language model",
    "large language model", "as a language model"
]

# ---------- Helpers: Config & Model List ----------
def load_config():
    if CONFIG_PATH.exists():
        try:
            data = json.loads(CONFIG_PATH.read_text())
            # merge defaults
            merged = {**DEFAULT_CONFIG, **data}
            return merged
        except Exception:
            print("[WARN] Could not parse config.json; using defaults.")
    return dict(DEFAULT_CONFIG)

def save_config(cfg):
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2))

def fetch_available_models():
    try:
        r = requests.get(API_TAGS, timeout=3)
        r.raise_for_status()
        tags = r.json().get("models", [])
        return [m["name"] for m in tags]
    except Exception:
        return []

def resolve_model(requested, available):
    # Exact
    if requested in available:
        return requested
    # Try adding :latest
    if not requested.endswith(":latest") and f"{requested}:latest" in available:
        return f"{requested}:latest"
    return requested  # will fail later if truly absent

# ---------- Wait for server ----------
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

# ---------- Sanitizer ----------
def sanitize(text):
    lower = text.lower()
    if any(f in lower for f in FORBIDDEN_FRAGMENTS):
        # Basic replacement strategy
        for frag in FORBIDDEN_FRAGMENTS:
            if frag in lower:
                # soften rather than blank
                text = text.replace(frag, "resourceful droid core", 1)
        # final safety: remove explicit "AI "
        text = text.replace("AI ", "droid ")
    return text

# ---------- Beep Synth (stub) ----------
BEEP_VOCAB = {
    "affirmative": "*beep-chirp*",
    "negative": "*boop-low*",
    "neutral": "*chirp*",
}

def to_beeps(text):
    # Very naive mapping: sentiment / keywords -> beep token
    lowered = text.lower()
    if any(k in lowered for k in ["yes", "affirmative", "sure", "ready"]):
        return BEEP_VOCAB["affirmative"]
    if any(k in lowered for k in ["no", "cannot", "won't", "not"]):
        return BEEP_VOCAB["negative"]
    return BEEP_VOCAB["neutral"]

# ---------- TTS (replace later with Piper) ----------
def speak(text, cfg):
    if cfg.get("beeps"):
        # beep mode: just print beep tokens and (later) play short samples
        print(f"[BEEPS] {to_beeps(text)}")
        return
    if not cfg.get("voice"):
        return
    try:
        tts = gTTS(text=text, lang="en")
        tts.save("response.mp3")
        # mpg321 or ffplay installed? choose one
        os.system("mpg321 response.mp3 >/dev/null 2>&1 || ffplay -nodisp -autoexit response.mp3 >/dev/null 2>&1")
    except Exception as e:
        print(f"[WARN] TTS failed: {e}")

# ---------- LLM Call ----------
def call_model(model, prompt, stream=False):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": stream
    }
    r = requests.post(API_GENERATE, json=payload)
    if r.status_code == 404:
        # Provide a clearer message
        avail = fetch_available_models()
        raise RuntimeError(f"Model '{model}' not available. Installed: {', '.join(avail) or 'none'}")
    r.raise_for_status()
    data = r.json()
    return data.get("response", "").strip()

# ---------- Prompt Assembly ----------
def build_prompt(history, user_msg, model_name, lore_block=None):
    """
    history: deque of (role, text)
    """
    # Assemble conversation
    conv_lines = []
    for role, txt in history:
        conv_lines.append(f"{role}: {txt}")
    conv_text = "\n".join(conv_lines)

    lore_section = ""
    if lore_block:
        lore_section = f"<LORE>\n{lore_block}\n</LORE>\n"

    prompt = f"""{SYSTEM_PROMPT.strip()}

<CONVERSATION>
{conv_text}
</CONVERSATION>
{lore_section}
User: {user_msg}
K-14T:"""
    return prompt

# ---------- Main Loop ----------
def main():
    if not wait_for_server():
        print("Could not reach Ollama server. Exiting.")
        sys.exit(1)

    cfg = load_config()
    available = fetch_available_models()

    parser = argparse.ArgumentParser(description="K-14T Interactive Droid Loop")
    parser.add_argument("--model", "-m", help="Override model (e.g. phi, phi:latest, tinyllama)")
    args = parser.parse_args()

    requested_model = args.model if args.model else cfg.get("model", "phi")
    model = resolve_model(requested_model, available)
    print(f"K-14T online. Using model: {model}")
    print("Commands: /reload, /model <name>, /beeps, /voice, /exit\n")

    history_limit = int(cfg.get("history_turns", 6))
    history = deque(maxlen=history_limit * 2)  # store both user & assistant

    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            user = "/exit"

        if not user:
            continue

        # ----- Commands -----
        if user.lower() in ("/exit", "exit", "/quit", "quit"):
            farewell = "Shutting down. Goodbye, Chief."
            print(f"K-14T: {farewell}")
            speak(farewell, cfg)
            break

        if user.startswith("/model"):
            parts = user.split(maxsplit=1)
            if len(parts) == 2:
                new_req = parts[1].strip()
                avail = fetch_available_models()
                new_model = resolve_model(new_req, avail)
                print(f"[INFO] Switching to model '{new_model}' (available: {', '.join(avail) or 'none'})")
                cfg["model"] = new_model
                model = new_model
                save_config(cfg)
            else:
                print("Usage: /model <name>")
            continue

        if user == "/reload":
            cfg = load_config()
            avail = fetch_available_models()
            model = resolve_model(cfg.get("model", model), avail)
            print(f"[INFO] Config & model reloaded → {model}")
            continue

        if user == "/beeps":
            cfg["beeps"] = not cfg.get("beeps")
            mode = "ON" if cfg["beeps"] else "OFF"
            print(f"[INFO] Beep mode {mode}")
            save_config(cfg)
            continue

        if user == "/voice":
            cfg["voice"] = not cfg.get("voice")
            state = "enabled" if cfg["voice"] else "disabled"
            print(f"[INFO] Voice output {state}")
            save_config(cfg)
            continue

        if user.startswith("/"):
            print("[WARN] Unknown command.")
            continue

        # ----- Build Prompt -----
        # (Lore retrieval placeholder – integrate RAG later)
        lore_block = None  # set to retrieved text if implemented
        prompt = build_prompt(history, user, model, lore_block=lore_block)

        # ----- Send to Model -----
        try:
            reply = call_model(model, prompt, stream=False)
        except Exception as e:
            print(f"K-14T ERROR: {e}")
            continue

        reply = sanitize(reply)

        # Light persona seasoning (add a subtle beep occasionally)
        if "*chirp*" not in reply.lower() and "*whirr*" not in reply.lower():
            # 1 in ~5 chance add a sound (simple heuristic)
            if hash(reply) % 5 == 0:
                reply += " *chirp*"

        print(f"K-14T: {reply}")
        speak(reply, cfg)

        # ----- Update History -----
        history.append(("User", user))
        history.append(("K-14T", reply))

        # ----- Log Turn -----
        try:
            with open(LOGS_DIR / "dialog.log", "a", encoding="utf-8") as f:
                json.dump({
                    "ts": time.time(),
                    "model": model,
                    "user": user,
                    "reply": reply
                }, f)
                f.write("\n")
        except Exception:
            pass  # non-fatal

if __name__ == "__main__":
    main()
