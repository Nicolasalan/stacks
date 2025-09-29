import torch
import random
from utils import tokenizador, obter_dataset
from transcricao import ModeloTranscricao

# --- Config (ajuste apenas o caminho do checkpoint se necessário) ---
CHECKPOINT = "/Users/nicolasalan/Documents/stacks/salvos/modelo_1200.pth"

# --- Inicializa tokenizador e dispositivo ---
tokenizadar = tokenizador()
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# --- Cria o modelo (mesma assinatura do treinamento) ---
modelo = ModeloTranscricao(
    num_codebooks=2,
    tamanho_codebooks=32,
    tamanho_incorporacao=16,
    num_transformers=2,
    tamanho_vocabulario=len(tokenizadar.get_vocab()),
    passos=[6, 6, 6],
    inicial_agrupamento_medio=4,
    max_comprimento_sequencia=400
).to(device)

# --- Carrega checkpoint (state_dict ou objeto salvo) ---
modelo.carregar(CHECKPOINT)

# --- Pega um exemplo aleatório do dataset processado ---
dataloader = obter_dataset(batch_size=1, num_exemplos=50, num_workers=0)
dataset = dataloader.dataset
idx = random.randint(0, len(dataset) - 1)
sample = dataset[idx]

print(sample)

audio = sample["audio"]
# se estiver multicanal, faz média simples
if audio.dim() == 2 and audio.shape[0] > 1:
    audio = audio.mean(dim=0)
audio = audio.unsqueeze(0).float().to(device)  # (1, seq_len)

# print("Ground truth:", sample.get("texto"))

# --- Inferência simples ---
with torch.no_grad():
    out, _ = modelo(audio)  # espera (B, T, V) em logits ou log_probs

# greedy + collapse + remove blank
pred_ids = torch.argmax(out, dim=-1)[0].cpu().numpy().tolist()
blank_id = tokenizadar.token_to_id("▀")

pred_tokens = []
prev = None
for tid in pred_ids:
    if tid != blank_id and tid != prev:
        pred_tokens.append(tokenizadar.id_to_token(tid))
    prev = tid

pred_text = "".join(pred_tokens)
# print("Transcrição predita:", pred_text)
