import torch.nn as nn
import torch

class QuantizadorVetorial(nn.Module):
    def __init__(self, num_incorporadores, tamanho_incorporadores, custo_compromisso=0.25):
        super().__init__()
        self.num_incorporadores = num_incorporadores
        self.tamanho_incorporadores = tamanho_incorporadores

        self.incorporador = nn.Embedding(num_incorporadores, tamanho_incorporadores)
        nn.init.uniform_(self.incorporador.weight,-0.1, 0.1)

        self.custo_compromisso = custo_compromisso

    def forward(self, x):
        batch_size, comprimento_sequencia, tamanho_incorporador = x.shape
        planar_x = x.reshape(batch_size * comprimento_sequencia, tamanho_incorporador)

        distancias = torch.cdist(planar_x, self.incorporador.weight, p=2)

        encoder_indices = torch.argmin(distancias, dim=1)
        quantizar = self.incorporador(encoder_indices).view(
            batch_size, comprimento_sequencia, tamanho_incorporador
        )

        e_perda_latente = torch.mean((quantizar.detach() - x)**2)
        q_perda_latente = torch.mean((quantizar - x.detach())**2)

        perda = q_perda_latente + self.custo_compromisso*e_perda_latente

        quantizar =x + (quantizar - x).detach()

        return quantizar, perda

class ResidualQuantizadorVetorial(nn.Module):
    def __init__(self, num_codebooks, tamanho_codebooks, tamanho_incorporador):
        super().__init__()
        self.codebooks = nn.ModuleList(
            [
                QuantizadorVetorial(tamanho_codebooks, tamanho_incorporador)
                for _ in range(num_codebooks)
            ]
        )

    def forward(self, x):
        saida = 0
        total_perda = 0
        for codebook in self.codebooks:
            temp_saida, temp_perda = codebook(x)
            x = x - temp_saida
            saida = saida+temp_saida
            total_perda += temp_perda

        return saida, total_perda

if __name__ == "__main__":
    rvq = ResidualQuantizadorVetorial(2, 16, 128)
    x = torch.randn(2, 12, 128, requires_grad=True)
    optimizer = torch.optim.Adam(rvq.parameters(), lr=0.005)

    for i in range(4):
        saida, vq_perda = rvq(x)

        recon_loss = torch.mean((saida - x)**2)
        total_perda = vq_perda + recon_loss

        total_perda.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(f"Iteracao {i}: Perda Total = {total_perda.item()}, VQ Perda = {vq_perda.item()}, Reconstrucao Perda = {recon_loss.item()}")
