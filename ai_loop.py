import requests
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