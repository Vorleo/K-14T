#!/usr/bin/env python3
import os, json, time, requests, sys
from memory_store import MemoryStore

# Optional: import Piper only if installed & desired
try:
    from tts_piper import PiperTTS
except ImportError:
    PiperTTS = None

CONFIG_FILE = "config.json"

DEFAULTS = {
    "model": "phi2",
    "voice_enabled": True,
    "fast_mode": False,
    "num_predict": 40,
    "temperature": 0.60,
    "top_p": 0.9,
    "callsign": "Joshua",
    "persona_line": "You are K-14T, a concise in-universe utility droid; answer Joshua briefly.",
    "piper_voice": "voices/en_US-lessac-high.onnx",
    "piper_config": "voices/en_US-lessac-high.onnx.json",
    "piper_length_scale": 0.92,
    "piper_noise_scale": 0.50,
    "piper_noise_w": 0.60,
    "piper_pitch_semitones": 2,
    "max_history_pairs": 2,
    "memory_max_injected": 2
}

def load_config():
    if not os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(DEFAULTS, f, indent=2)
        return dict(DEFAULTS)
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    for k,v in DEFAULTS.items():
        data.setdefault(k,v)
    return data

cfg = load_config()

API = "http://localhost:11434/api/generate"
history = []  # list of (role, text) role in {"User","K-14T"}
memory = MemoryStore()

# TTS
tts = None
if cfg.get("voice_enabled") and PiperTTS:
    tts = PiperTTS(
        cfg["piper_voice"],
        cfg["piper_config"],
        cfg["piper_length_scale"],
        cfg["piper_noise_scale"],
        cfg["piper_noise_w"],
        cfg["piper_pitch_semitones"]
    )
    if tts and not tts.enabled:
        print("[VOICE] Disabled:", getattr(tts, "error", "unknown error"))
        tts = None

def save_config():
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

def build_prompt(user_text: str):
    # Retrieve memory facts
    mem_facts = memory.retrieve(user_text, cfg["memory_max_injected"])
    mem_line = ""
    if mem_facts:
        mem_line = "[MEM] " + " | ".join(mem_facts)
    # Trim history
    max_pairs = cfg["max_history_pairs"]
    trimmed = history[-(max_pairs*2):]
    lines = [cfg["persona_line"]]
    if mem_line:
        lines.append(mem_line)
    for role, txt in trimmed:
        lines.append(f"{role}: {txt}")
    lines.append(f"User: {user_text}")
    lines.append("K-14T:")
    return "\n".join(lines)

def generate(prompt: str):
    opts = {
        "num_predict": cfg["num_predict"],
        "temperature": cfg["temperature"],
        "top_p": cfg["top_p"]
    }
    payload = {
        "model": cfg["model"],
        "prompt": prompt,
        "stream": False,
        "options": opts
    }
    t0 = time.time()
    r = requests.post(API, json=payload)
    t1 = time.time()
    if not r.ok:
        raise RuntimeError(f"Model error: {r.status_code} {r.text[:120]}")
    data = r.json()
    raw = data.get("response", "").strip()
    dur = t1 - t0
    print(f"[TIMING] {dur:.2f}s (model={cfg['model']} tokens≈{len(raw.split())})")
    return raw

def postprocess(text: str):
    # Keep only first sentence to stay snappy / short
    cut_chars = ".!?。"
    for i, ch in enumerate(text):
        if ch in cut_chars:
            return text[:i+1].strip()
    return text.strip()

def cmd_status():
    print(f"Model={cfg['model']} voice={'ON' if cfg['voice_enabled'] and tts else 'OFF'} "
          f"fast_mode={cfg['fast_mode']} num_predict={cfg['num_predict']} "
          f"history_pairs={cfg['max_history_pairs']}")

def toggle_fast():
    cfg['fast_mode'] = not cfg['fast_mode']
    if cfg['fast_mode']:
        cfg['num_predict'] = min(cfg['num_predict'], 35)
        cfg['voice_enabled'] = False
        print("Fast mode ON (num_predict<=35, voice off).")
    else:
        cfg['num_predict'] = DEFAULTS['num_predict']
        cfg['voice_enabled'] = True
        print("Fast mode OFF (restored defaults).")
    save_config()

def toggle_voice():
    global tts
    cfg['voice_enabled'] = not cfg['voice_enabled']
    if cfg['voice_enabled']:
        if PiperTTS:
            tts = PiperTTS(
                cfg["piper_voice"],
                cfg["piper_config"],
                cfg["piper_length_scale"],
                cfg["piper_noise_scale"],
                cfg["piper_noise_w"],
                cfg["piper_pitch_semitones"]
            )
            if tts and not tts.enabled:
                print("[VOICE] Could not enable:", getattr(tts, "error", "error"))
                cfg['voice_enabled'] = False
        else:
            print("[VOICE] piper-tts not installed.")
            cfg['voice_enabled'] = False
    else:
        tts = None
    save_config()
    print(f"Voice now {'ON' if cfg['voice_enabled'] else 'OFF'}.")

def switch_model(new_name: str):
    cfg['model'] = new_name
    save_config()
    print("Switched model ->", new_name)

def main():
    print("K-14T interactive. Commands: /remember, /mem, /forget <i>, /model <name>, /voice, /fast, /status, /exit")
    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[EXIT]")
            break
        if not user:
            continue
        if user.startswith("/"):
            parts = user.split()
            cmd = parts[0].lower()
            if cmd in ("/exit","/quit"):
                break
            elif cmd == "/remember":
                fact = " ".join(parts[1:]).strip()
                if fact:
                    memory.add_fact(fact)
                    print("K-14T: Stored.")
                else:
                    print("K-14T: Provide text.")
            elif cmd == "/mem":
                facts = memory.list_facts(20)
                if not facts:
                    print("K-14T: (no facts)")
                else:
                    for i, fct in enumerate(facts):
                        print(f"[{i}] {fct['fact']}")
            elif cmd == "/forget":
                if len(parts) < 2 or not parts[1].isdigit():
                    print("Usage: /forget <index>")
                else:
                    ok = memory.delete_index(int(parts[1]))
                    print("K-14T: Deleted." if ok else "K-14T: Bad index.")
            elif cmd == "/model":
                if len(parts) < 2:
                    print("Usage: /model <name>")
                else:
                    switch_model(parts[1])
            elif cmd == "/voice":
                toggle_voice()
            elif cmd == "/fast":
                toggle_fast()
            elif cmd == "/status":
                cmd_status()
            else:
                print("K-14T: Unknown command.")
            continue

        prompt = build_prompt(user)
        try:
            raw = generate(prompt)
        except Exception as e:
            print("K-14T: Error contacting model.", e)
            continue
        reply = postprocess(raw)
        # Simple guard: remove leading role tags
        if reply.lower().startswith("user:"):
            reply = reply.split(":",1)[-1].strip()
        if reply.lower().startswith("k-14t:"):
            reply = reply.split(":",1)[-1].strip()
        print(f"K-14T: {reply}")
        if cfg.get("voice_enabled") and tts and tts.enabled:
            tts.speak(reply)
        history.append(("User", user))
        history.append(("K-14T", reply))

    print("K-14T: Shutdown.")

if __name__ == "__main__":
    main()
