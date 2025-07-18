import os
import json
import argparse
import requests
from gtts import gTTS
import subprocess
from datetime import datetime
from pathlib import Path

API_URL = "http://localhost:11434/api/generate"
CONFIG_PATH = Path(__file__).parent / "config.json"

def load_config():
    if not CONFIG_PATH.exists():
        print(f"⚠️  config.json not found at {CONFIG_PATH}. Using defaults.")
        return {
            "model": "phi",
            "voice_enabled": True,
            "beep_mode": False,
            "system_prompt": "",
            "max_reply_chars": 600,
            "temperature": 0.7,
            "memory_file": "memory/log.txt",
            "auto_save_memory": True
        }
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            print("⚠️  config.json is invalid JSON:", e)
            return {}
    return data

def ensure_server():
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None

def available_models():
    tags = ensure_server()
    if not tags:
        return []
    # Response structure: {"models":[{"name":"phi"}...]}
    return [m.get("name") for m in tags.get("models", [])]

def get_response(model, prompt, temperature=0.7, system_prompt=""):
    # Some smaller models ignore temperature, but we include it for future-proofing
    full_prompt = f"{system_prompt.strip()}\n\nUser: {prompt}\nK-14T:"
    payload = {
        "model": model,
        "prompt": full_prompt,
        "stream": False
    }
    r = requests.post(API_URL, json=payload, timeout=180)
    r.raise_for_status()
    return r.json().get("response", "").strip()

def speak(text):
    tts = gTTS(text=text, lang="en")
    tts.save("response.mp3")
    subprocess.run(["mpg321", "response.mp3"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def save_memory(cfg, user, reply):
    if not cfg.get("auto_save_memory", True):
        return
    mem_path = Path(cfg.get("memory_file", "memory/log.txt"))
    mem_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with mem_path.open("a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] USER: {user}\n[{timestamp}] K-14T: {reply}\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Override model name from config.json")
    args = parser.parse_args()

    cfg = load_config()

    # Allow override via CLI or environment variable
    model = args.model or os.getenv("K14T_MODEL") or cfg.get("model", "phi")

    models_installed = available_models()
    if models_installed and model not in models_installed:
        print(f"⚠️  Model '{model}' not installed. Installed: {models_installed}")
        if cfg.get("model") in models_installed:
            model = cfg.get("model")
            print(f"→ Falling back to config model: {model}")
        elif "phi" in models_installed:
            model = "phi"
            print(f"→ Falling back to 'phi'")
        else:
            print("❌ No usable model found. Install one with: ollama pull phi")
            return

    print(f"K-14T online. Using model: {model}")
    print("Commands: /reload (reload config), /model <name>, /exit to quit.\n")

    while True:
        user = input("You: ").strip()
        if not user:
            continue

        if user.startswith("/"):
            parts = user.split()
            cmd = parts[0].lower()
            if cmd in ("/exit", "/quit", "/bye"):
                print("K-14T: Shutting down. Goodbye.")
                if cfg.get("voice_enabled", True):
                    speak("Goodbye")
                break
            elif cmd == "/reload":
                cfg = load_config()
                print("🔄 Config reloaded.")
                continue
            elif cmd == "/model":
                if len(parts) < 2:
                    print("Usage: /model <installed_model_name>")
                    continue
                req_model = parts[1]
                if req_model in available_models():
                    model = req_model
                    print(f"✅ Switched to model: {model}")
                else:
                    print(f"⚠️ Model '{req_model}' not installed.")
                continue
            else:
                print("Unknown command.")
                continue

        try:
            reply = get_response(
                model=model,
                prompt=user,
                temperature=cfg.get("temperature", 0.7),
                system_prompt=cfg.get("system_prompt", "")
            )
        except Exception as e:
            print("Error contacting model:", e)
            continue

        # Truncate if too long
        max_chars = cfg.get("max_reply_chars", 600)
        if len(reply) > max_chars:
            reply = reply[:max_chars] + " …"

        print("K-14T:", reply)
        save_memory(cfg, user, reply)
        if cfg.get("voice_enabled", True) and not cfg.get("beep_mode", False):
            speak(reply)
        # Future: if beep_mode True, call a beep renderer instead

if __name__ == "__main__":
    main()


