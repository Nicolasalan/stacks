import torch
import torch.nn as nn
import math
import torch.nn.functional as F

# calculo da atencao do calcular_atencao ate FeedFoward
def calcular_atencao(
    valor: torch.Tensor,
    chave: torch.Tensor,
    consulta: torch.Tensor
):
    pontuacao_atencao = torch.matmul(consulta, chave.transpose(-2, -1))
    pontuacao_atencao = pontuacao_atencao / math.sqrt(chave.shape[-1])
    pontuacao_atencao = F.softmax(pontuacao_atencao, dim=-1)
    atencao = torch.matmul(pontuacao_atencao, valor)
    return atencao, pontuacao_atencao

class FeedFoward(nn.Module):
    def __init__(self, tamanho_incorporcao: int):
        super().__init__()
        self.camada1 = nn.Linear(tamanho_incorporcao, tamanho_incorporcao)
        self.camada2 = nn.Linear(tamanho_incorporcao, tamanho_incorporcao)

    def forward(self, x):
        x = self.camada1(x)
        x = F.gelu(x)
        x = self.camada2(x)
        return x

# basicmanete ele vai pegar os embedding anteriores junto com o atual e decidir qual é mais importante
class CamadaAtencao(nn.Module):
    def __init__(self, tamanho_incorporcao: int):
        super().__init__()
        self.tamanho_incorporcao =tamanho_incorporcao
        self.consulta_denso = nn.Linear(tamanho_incorporcao, tamanho_incorporcao)
        self.chave_denso = nn.Linear(tamanho_incorporcao, tamanho_incorporcao)
        self.valor_denso = nn.Linear(tamanho_incorporcao, tamanho_incorporcao)

    def forward(self, incorporacao: torch.Tensor):
        consulta = self.consulta_denso(incorporacao)
        chave = self.chave_denso(incorporacao)
        valor = self.valor_denso(incorporacao)
        atencao, _ = calcular_atencao(valor, chave, consulta)
        return atencao

class BlocoAtencao(nn.Module):
    def __init__(self, tamanho_incorporacao: int):
        super().__init__()
        self.camada_atencao = CamadaAtencao(tamanho_incorporacao)
        self.feed_forward = FeedFoward(tamanho_incorporacao)
        self.camada_normalizar = nn.LayerNorm(tamanho_incorporacao)

    def forward(self, x):
        contexto = self.camada_atencao(x)
        contexto = self.camada_normalizar(contexto)
        contexto =  self.feed_forward(contexto)
        contexto = F.gelu(contexto)
        saida = contexto + x
        return saida

class Transformer(nn.Module):
    def __init__(self, tamanho_incorporacao, num_camadas, maximo_comprimento_sequencia):
        super().__init__()
        self.encoder_posicional = CodificacaoPosicaoSenoidal(
            tamanho_incorporacao, maximo_comprimento_sequencia
        )
        self.bloco_transformers = nn.ModuleList(
            [BlocoAtencao(tamanho_incorporacao) for _ in range(num_camadas)]
        )

    def forward(self, x):
        x = self.encoder_posicional(x)
        for bloco in self.bloco_transformers:
            x = bloco(x)
        return x

class MultiCabecaAtencao(nn.Module):
    def __init__(self, tamanho_incorporacao, num_cabecas):
        super().__init__()
        assert (
            tamanho_incorporacao % num_cabecas == 0
        ), "Tamanho da incorporação deve ser divisível pelo número de cabeças"

        self.tamanho_incorporacao = tamanho_incorporacao
        self.num_cabecas = num_cabecas
        self.tamanho_cabecas = tamanho_incorporacao // num_cabecas
        self.consulta = nn.Linear(tamanho_incorporacao, tamanho_incorporacao)
        self.chave = nn.Linear(tamanho_incorporacao, tamanho_incorporacao)
        self.valor = nn.Linear(tamanho_incorporacao, tamanho_incorporacao)
        self.saida_linear = nn.Linear(tamanho_incorporacao, tamanho_incorporacao)

    def forward(self, incorporacao):
        consulta = self.consulta(incorporacao)
        chave = self.chave(incorporacao)
        valor = self.valor(incorporacao)

        atencao = consulta @ chave.transpose(-2, -1) / math.sqrt(self.tamanho_cabecas)
        atencao = F.softmax(atencao, dim=-1)
        atencao = atencao @ valor

        saida = self.saida_linear(atencao)
        return saida


class CodificacaoPosicaoSenoidal(nn.Module):
    def __init__(self, tamanho_incorporacao: int, maximo_comprimento_sequencia: int):
        super().__init__()
        posicao = torch.arange(maximo_comprimento_sequencia).unsqueeze(1)
        termo_divisao = torch.exp(torch.arange(0, tamanho_incorporacao, 2) * (-math.log(10000.0) / tamanho_incorporacao))

        pe = torch.zeros(maximo_comprimento_sequencia, tamanho_incorporacao)
        pe[:, 0::2] = torch.sin(posicao * termo_divisao)
        pe[:, 1::2] = torch.cos(posicao * termo_divisao)

        self.register_buffer("positional_embedding", pe)

    def forward(self, x):
        return x + self.positional_embedding[: x.size(1), :]


if __name__ == "__main__":
    transformers = Transformer(
        tamanho_incorporacao=128,
        num_camadas=3,
        maximo_comprimento_sequencia=512
    )
    x = torch.randn(2, 10, 128)
    print(transformers(x).shape)
