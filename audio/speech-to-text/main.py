import torch

# dataset = datasets.load_dataset(
#     "m-aliabbas/idrak_timit_subsample1", split="train")

# dataset = dataset.cast_column("audio", datasets.Audio(sampling_rate=None))

# data = next(iter(dataset))
# print(data)
# # possui uma coluna de array (uma matriz), transcricao e sample rate
# audio = data["audio"]["array"]
# sample_rate = data["audio"]["sampling_rate"]
# transcript = data["transcription"]

# print(audio.shape) # valor escalar de 35840
# print(sample_rate) # quantas amplitudes captadas em 1 segundo
# print("Tamanho do audio: ", audio.size / sample_rate)
# print(transcript)


# plotar o audio
import matplotlib.pyplot as plt
import numpy as np

# comeco_ms = 0
# fim_ms = 1000

# comeco_idx = int(comeco_ms * sample_rate / 1000)
# fim_idx = int(fim_ms * sample_rate / 1000)

# print(fim_idx)

# audio_numpy = audio[comeco_idx:fim_idx]

# print(audio_numpy)

# tempo_em_ms = np.arange(len(audio_numpy)) * (1000 / sample_rate)

# plt.figure(figsize=(15, 5))
# plt.plot(tempo_em_ms, audio_numpy)
# plt.xlabel("Tempo em ms")
# plt.ylabel("Amplitude")
# plt.grid(True)
# plt.show()




# if __name__ == "__main__":
    # dataloader = obter_dataset(batch_size=8)
    # for batch in dataloader:
    #     audio = batch["audio"]
    #     entradas_ids = batch["entrada_ids"]
    #     print(audio.shape) # entrada de audio
    #     print(entradas_ids.shape) # entrada de texto ?

    #     # breakpoint()
    #     break
