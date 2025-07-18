import requests
import os
import argparse
import requests
from gtts import gTTS
import subprocess

API_URL = "http://localhost:11434/api/generate"

def ensure_server():
    # Quick probe to see if ollama server responds; simple retry
    for _ in range(5):
        try:
            r = requests.get("http://localhost:11434/api/tags", timeout=2)
            if r.status_code == 200:
                return True
        except Exception:
            pass
    return False

def get_response(model, prompt):
    r = requests.post(API_URL,
                      json={"model": model, "prompt": prompt, "stream": False},
                      timeout=120)
    r.raise_for_status()
    return r.json().get("response", "").strip()

def speak(text):
    tts = gTTS(text=text, lang="en")
    tts.save("response.mp3")
    subprocess.run(["mpg321", "response.mp3"],
                   stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.getenv("K14T_MODEL", "phi"),
                        help="Model name installed in Ollama (e.g. phi, tinyllama, mistral)")
    args = parser.parse_args()

    model = args.model

    if not ensure_server():
        print("⚠️ Ollama server not responding. Is the service running?")
        return

    print(f"K-14T online. Using model: {model}. Type 'exit' to quit.\n")

    while True:
        user = input("You: ").strip()
        if user.lower() in ("exit", "quit", "bye"):
            print("K-14T: Shutting down. Goodbye.")
            speak("Goodbye")
            break

        try:
            resp = get_response(model, user)
        except Exception as e:
            print("Error talking to model:", e)
            continue

        print("K-14T:", resp)
        speak(resp)

if __name__ == "__main__":
    main()
import os
from gtts import gTTS

API_URL = "http://localhost:11434/api/generate"

def get_response(prompt):
    r = requests.post(API_URL, json={
        "model": "phi",
        "prompt": prompt,
        "stream": False
    })
    r.raise_for_status()
    return r.json().get("response", "").strip()

def speak(text):
    tts = gTTS(text=text, lang="en")
    tts.save("response.mp3")
    os.system("mpg321 response.mp3 >/dev/null 2>&1")

def main():
    print("K-14T is online. Type 'exit' to quit.\n")
    while True:
        user = input("You: ").strip()
        if user.lower() in ("exit","quit","bye"):
            print("K-14T: Goodbye!")
            speak("Goodbye")
            break
        resp = get_response(user)
        print("K-14T:", resp)
        speak(resp)

if __name__ == "__main__":
    main()
