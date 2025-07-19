import os
from piper import PiperVoice
import soundfile as sf
import subprocess
import tempfile

class PiperTTS:
    def __init__(self, model_path, cfg_path, length_scale=1.0,
                 noise_scale=0.5, noise_w=0.6, pitch_semitones=0):
        self.enabled = False
        if not (os.path.exists(model_path) and os.path.exists(cfg_path)):
            self.error = f"Missing voice files: {model_path} / {cfg_path}"
            return
        try:
            self.voice = PiperVoice.load(model_path, cfg_path)
            self.length_scale = length_scale
            self.noise_scale = noise_scale
            self.noise_w = noise_w
            self.pitch_semitones = pitch_semitones
            self.enabled = True
        except Exception as e:
            self.error = str(e)

    def speak(self, text:str):
        if not self.enabled or not text.strip():
            return
        audio = self.voice.synthesize(
            text,
            length_scale=self.length_scale,
            noise_scale=self.noise_scale,
            noise_w=self.noise_w
        )
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, audio, self.voice.sample_rate)
            wav_path = tmp.name
        # Apply simple pitch / speed with sox if pitch needed
        final_path = wav_path
        if self.pitch_semitones:
            adj = wav_path + "_p.wav"
            subprocess.run([
                "sox", wav_path, adj, "pitch", str(self.pitch_semitones*100)
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            final_path = adj
        subprocess.run(["aplay", final_path],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # cleanup
        try:
            os.remove(wav_path)
            if final_path != wav_path:
                os.remove(final_path)
        except:
            pass
