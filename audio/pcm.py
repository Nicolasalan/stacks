import soundfile as sf
import numpy as np

# Parâmetros do áudio
sample_rate = 44100
channels = 1

# Carregar PCM cru
with open("/Users/nicolasalan/Documents/stacks/audio/audio_20250915_131718.pcm", "rb") as f:
    pcm_data = f.read()

# Converter bytes para int16
audio_data = np.frombuffer(pcm_data, dtype=np.int16)

# Salvar como WAV
sf.write("audio.wav", audio_data, sample_rate, subtype='PCM_16')
print("Arquivo WAV criado: audio.wav")
