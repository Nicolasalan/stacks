import torch.nn.functional as F
import torch
import torch.nn as nn

class BlocoDeReamostragemResidual(nn.Module):
    def __init__(self, entrada_canais, saida_canais, passo, tamanho_kernel=4):
        super().__init__()
        self.conv1 = nn.Conv1d(
            entrada_canais, saida_canais, kernel_size=tamanho_kernel, padding="same"
        )
        self.bn1 = nn.BatchNorm1d(saida_canais) # mudanca covariavel interna
        self.conv2 = nn.Conv1d(
            saida_canais,
            saida_canais,
            kernel_size=tamanho_kernel,
            stride=passo,
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (batch_size, in_channels, seq_len)
        saida = self.conv1(x)
        saida = self.bn1(saida)
        saida = self.relu(saida) + x # aqui que entra o residual (aprende o erro)
        saida = self.conv2(saida)
        return saida

class RedeReducaoAmostragem(nn.Module):
    def __init__(
        self,
        tamanho_incorporacao=128, # numero de features (embedding dim)
        tamanho_camada_oculta=64,
        entrada_canais=1,
        inicial_agrupamento_medio=2,
        passos=[6, 6, 8, 4, 2], # aqui sao as reducoes, no caso sao 5 reducoes com [6x, 6x, ...]
    ):
        super().__init__()
        self.camadas_densas = nn.ModuleList()
        self.agrupamento_medio = nn.MaxPool1d(kernel_size=inicial_agrupamento_medio, padding=1)

        for i in range(len(passos)):
            self.camadas_densas.append(
                BlocoDeReamostragemResidual(
                    tamanho_camada_oculta if i > 0 else entrada_canais,
                    tamanho_camada_oculta,
                    passos[i],
                    tamanho_kernel=8,
                )
            )
        self.final_conv = nn.Conv1d(
            tamanho_camada_oculta, tamanho_incorporacao, kernel_size=4, padding="same",
        )

    def forward(self, x):
        # print("shape: ", x.shape)
        x = self.agrupamento_medio(x)
        # print("Mean shape: ", x.shape)
        for i, camada in enumerate(self.camadas_densas):
            x = camada(x)
            # print(f"Camada {i} shape: ", x.shape)

        x = self.final_conv(x)
        # print("shape final: ", x.shape)
        x = x.transpose(1, 2)
        # print("shape transpose: ", x.shape)
        return x

# Teste
if __name__ == "__main__":
    batch_size = 2
    entrada_canais = 1
    seq_len = 237680
    saida_incorporacao =32
    camada_oculta = 16
    passos = [2, 4, 8]
    agrupamento_medio = 2

    rede_redutor = RedeReducaoAmostragem(
        tamanho_incorporacao=saida_incorporacao, # numero de features (embedding dim)
        tamanho_camada_oculta=camada_oculta,
        entrada_canais=entrada_canais,
        inicial_agrupamento_medio=agrupamento_medio,
        passos=passos
    )
    x = torch.randn(batch_size, entrada_canais, seq_len)
    print(rede_redutor(x).shape)

    # torch.Size([2, 16, 59417])
    # batch size, camada oculta, dados
