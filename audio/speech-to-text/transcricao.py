import torch.nn as nn
import torch
from rvq import ResidualQuantizadorVetorial, QuantizadorVetorial

from atencao import Transformer
from residual import RedeReducaoAmostragem

class ModeloTranscricao(nn.Module):
    def __init__(
        self,
        num_codebooks: int,
        tamanho_codebooks: int,
        tamanho_incorporacao: int,
        tamanho_vocabulario: int,
        passos: list[int],
        inicial_agrupamento_medio: int,
        num_transformers: int,
        max_comprimento_sequencia: int = 2000,
    ):
        super().__init__()
        self.opcoes = {
            'num_codebooks': num_codebooks,
            'tamanho_codebooks': tamanho_codebooks,
            'tamanho_incorporacao': tamanho_incorporacao,
            'tamanho_vocabulario': tamanho_vocabulario,
            'passos': passos,
            'inicial_agrupamento_medio': inicial_agrupamento_medio,
            'num_transformers': num_transformers,
            'max_comprimento_sequencia': max_comprimento_sequencia
        }

        self.reducao_amostragem = RedeReducaoAmostragem(
            tamanho_incorporacao=tamanho_incorporacao,
            tamanho_camada_oculta=tamanho_incorporacao // 2,
            passos=passos,
            inicial_agrupamento_medio=inicial_agrupamento_medio
        )

        self.pre_rvq_com_transformers = Transformer(
            tamanho_incorporacao=tamanho_incorporacao,
            num_camadas=num_transformers,
            maximo_comprimento_sequencia=max_comprimento_sequencia,
        )

        self.rvq = ResidualQuantizadorVetorial(
            num_codebooks=num_codebooks,
            tamanho_codebooks=tamanho_codebooks,
            tamanho_incorporador=tamanho_incorporacao,
        )

        self.camada_final = nn.Linear(tamanho_incorporacao, tamanho_vocabulario)

    def forward(self, x):
        perda = torch.tensor(0.0)
        x = x.unsqueeze(1)
        x = self.reducao_amostragem(x)
        x = self.pre_rvq_com_transformers(x)
        x, perda = self.rvq(x)
        x = self.camada_final(x)
        x = torch.log_softmax(x, dim=-1)
        return x, perda

    def salvar(self, caminho):
        torch.save({"modelo": self.state_dict(), "opcoes": self.opcoes}, caminho)

    @staticmethod
    def carregar(caminho):
        modelo = ModeloTranscricao(**torch.load(caminho)["opcoes"])
        modelo.load_state_dict(torch.load(caminho)["modelo"])
        return modelo

if __name__ == "__main__":
    modelo = ModeloTranscricao(
        num_codebooks=3,
        tamanho_codebooks=64,
        tamanho_incorporacao=64,
        tamanho_vocabulario=30,
        passos=[6, 8, 4, 2],
        inicial_agrupamento_medio=4,
        max_comprimento_sequencia=2000,
        num_transformers=2
    )
    x = torch.randn(4, 237680)
    saida, perda = modelo(x)
    print(saida.shape)
