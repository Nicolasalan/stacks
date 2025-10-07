import torch
import random
from utils import tokenizador, obter_dataset
from transcricao import ModeloTranscricao

# --- Config (ajuste apenas o caminho do checkpoint se necessário) ---
CHECKPOINT = "/Users/nicolasalan/Projetos/stacks/salvos/modelo_1200.pth"

# --- Inicializa tokenizador e dispositivo ---
tokenizadar = tokenizador()
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# --- Carrega o modelo e as opções do checkpoint ---
print(f"Carregando modelo do checkpoint: {CHECKPOINT}")
modelo = ModeloTranscricao.carregar(CHECKPOINT).to(device)
modelo.eval()

# --- Pega um exemplo aleatório do dataset processado ---
dataloader = obter_dataset(batch_size=1, num_exemplos=50, num_workers=0)
dataset = dataloader.dataset
idx = random.randint(0, len(dataset) - 1)
sample = dataset[idx]

audio = sample["audio"]
# se estiver multicanal, faz média simples
if audio.dim() == 2 and audio.shape[0] > 1:
    audio = audio.mean(dim=0)
audio = audio.unsqueeze(0).float().to(device)  # (1, seq_len)

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
print("Texto Original:     ", sample.get("texto"))
print("Transcrição Predita:", pred_text)
