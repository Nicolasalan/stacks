import os

if not os.path.exists("salvos"):
    os.makedirs("salvos")

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
torch.autograd.set_detect_anomaly(True)
from utils import obter_dataset, tokenizador
from transcricao import ModeloTranscricao
from torch import nn

vq_inicial_pesos_perda = 10
vq_aquecimento_passo = 1000
vq_final_perda_peso = 0.5
num_epocas = 1000
num_exemplos = None
id_modelo = "test"
num_batch_repete = 1

comeco_passo = 0
BATCH_SIZE = 64
LR = 0.005

def executar_perda(log_probs, alvo, token_branco):

    funca_perda = nn.CTCLoss(blank=token_branco)
    tamanho_entrada = tuple(log_probs.shape[1] for _ in range(log_probs.shape[0]))
    tamanho_alvo =  (alvo != token_branco).sum(dim=1)
    tamanho_alvo = tuple(t.item() for t in tamanho_alvo)
    entrada_primeira_sequencia = log_probs.permute(1, 0, 2)

    perda = funca_perda(entrada_primeira_sequencia, alvo, tamanho_entrada, tamanho_alvo)

    return perda

def main():
    diretorio_registro = f"runs/falap2texto_treinamento/{id_modelo}"
    if os.path.exists(diretorio_registro):
        import shutil

        shutil.rmtree(diretorio_registro)

    tokenizadar = tokenizador()
    token_branco = tokenizadar.token_to_id("▀")

    dispositivo = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    print("Usando dispositivo:", dispositivo)

    if os.path.exists(f"modelos/{id_modelo}/ultimo_modelo.pth"):
        print("Carregando modelo ...")
        modelo = ModeloTranscricao.load(f"modelos/{id_modelo}/ultimo_modelo.pth").to(dispositivo)
        modelo.eval()
    else:
        modelo = ModeloTranscricao(
            num_codebooks=2,
            tamanho_codebooks=64,
            tamanho_incorporacao=64,
            num_transformers=2,
            tamanho_vocabulario=len(tokenizadar.get_vocab()),
            passos=[6, 6, 6],
            inicial_agrupamento_medio=4,
            max_comprimento_sequencia=400
        ).to(dispositivo)

    numero_parametros_treinados = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    print(f"Número de parâmetros treinados: {numero_parametros_treinados}")

    otimizador = torch.optim.Adam(modelo.parameters(), lr=LR)

    dataloader = obter_dataset(
        batch_size=BATCH_SIZE,
        num_exemplos=num_exemplos,
        num_workers=0  # Mudança para 0 para evitar problemas de multiprocessing no macOS
    )

    ctc_perdas = []
    vq_perdas = []
    num_batches = len(dataloader)
    passos = comeco_passo

    for i in range(num_epocas):
        for idx, batch in enumerate(dataloader):
            for repetir_batch in range(num_batch_repete):
                audio = batch["audio"]
                alvo = batch["entrada_ids"]
                texto = batch["texto"]

                if alvo.shape[1] > audio.shape[1]:
                    print(
                        "Preenchimento audio, alvo é maior que o audio. Tamanho do audio: ",
                        audio.shape,
                        "Tamanho do alvo: ",
                        alvo.shape
                    )
                    audio = torch.nn.functional.pad(
                        audio, (0, 0, 0, alvo.shape[1] - audio.shape[1])
                    )
                    print("Depois do preenchimento: ", audio.shape)

                audio = audio.to(dispositivo)
                alvo = alvo.to(dispositivo)

                otimizador.zero_grad()
                saida, vq_perda = modelo(audio)
                ctc_perda = executar_perda(saida, alvo, token_branco)

                vq_perda_peso =  max(
                    vq_final_perda_peso,
                    vq_inicial_pesos_perda
                    - (vq_inicial_pesos_perda - vq_final_perda_peso)
                    * (passos / vq_aquecimento_passo)
                )

                if vq_perda is None:
                    perda = ctc_perda

                else:
                    perda = ctc_perda + vq_perda_peso * vq_perda

                if torch.isinf(perda):
                    print("Perda fico infinita carai ")
                    continue
                perda.backward()

                torch.nn.utils.clip_grad_norm_(
                    modelo.parameters(),
                    max_norm=10.0
                )

                otimizador.step()

                ctc_perdas.append(ctc_perda.item())
                vq_perdas.append(vq_perda.item())
                passos += 1

                if passos % 20 == 0:
                    media_ctc_perda = sum(ctc_perdas) / len(ctc_perdas)
                    media_vq_perda = sum(vq_perdas) / len(vq_perdas)
                    media_perda = media_ctc_perda + vq_perda_peso * media_vq_perda
                    print(f"Passo {passos}: CTC Loss = {media_ctc_perda:.4f}, VQ Loss = {media_vq_perda:.4f}, Total Loss = {media_perda:.4f}")

                # Salvar o modelo a cada 100 passos
                if passos % 200 == 0:
                    modelo.salvar(f"salvos/modelo_{passos}.pth")

                if passos % 20 == 0:
                    modelo.eval()
                    with torch.no_grad():
                        audio_aleatorio = audio[1]
                        texto_aleatorio = texto[1]
                        texto_tokenizado = tokenizadar.encode(texto_aleatorio)
                        x = audio_aleatorio.unsqueeze(0)
                        saida, _ = modelo(x)
                        ids_preditos = torch.argmax(saida, dim=-1)[0].cpu().numpy().tolist()
                        token_preditos = []
                        for token in ids_preditos:
                            token_preditos.append(tokenizadar.id_to_token(token))

                        print(f"Texto Predito: {token_preditos}, \n Texto verdadeiro: {texto_aleatorio} \n")

if __name__ == "__main__":
    main()
