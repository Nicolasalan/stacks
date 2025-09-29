import datasets
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

def collate_fn(batch):
    max_audio_len = max([item["audio"].shape[0] for item in batch])
    max_ids_len = 0
    has_entrada_ids = "entrada_ids" in batch[0]
    if has_entrada_ids:
        max_ids_len = max([len(item["entrada_ids"]) for item in batch])

    audio_tensor = torch.stack(
        [
            F.pad(item["audio"], (0, max_audio_len - item["audio"].shape[0]))
            for item in batch
        ]
    )
    output_dict= {
        "audio": audio_tensor,
        "texto": [item["texto"] for item in batch],
    }

    if has_entrada_ids:
        entrada_ids = torch.stack(
            [
                F.pad(
                    torch.tensor(item["entrada_ids"]),
                    (0, max_ids_len - len(item["entrada_ids"])),
                    value=0
                )
                for item in batch
            ]
        )

        output_dict["entrada_ids"] = entrada_ids

    return output_dict


def tokenizador(caminho_arquivo="tokenizador.json"):
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders
    
    tokenizar = Tokenizer(models.BPE())
    tokenizar.add_special_tokens(["▀"]) # token em branco
    tokenizar.add_tokens(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ '")) # alfabet + '
    tokenizar.pre_tokenizer = pre_tokenizers.ByteLevel()
    tokenizar.decoder = decoders.ByteLevel()
    tokenizar.blank_token = tokenizar.token_to_id("")
    tokenizar.save(caminho_arquivo)

    return tokenizar

# tokenizadordron = tokenizador()

# print(tokenizadordron.get_vocab())

# transcricao = transcript.upper()
# entrada_ids = tokenizadordron.encode(transcricao)

# print(entrada_ids) # ou entrada_ids.ids


class DatasetAudio(Dataset):

    def __init__(
        self,
        dataset,
        num_examplos=None,
        tokenizador=None
    ):
        self.dataset = dataset
        self.num_examplos = (
            min(num_examplos, len(dataset))
            if num_examplos is not None
            else len(dataset)
        )
        self.tokenizador = tokenizador

    def __len__(self):
        return self.num_examplos

    def __getitem__(self, idx):
        item = self.dataset[idx]
        waveform =torch.from_numpy(item["audio"]["array"]).float()
        texto = item["transcription"].upper()
        if self.tokenizador:
            encoder = self.tokenizador.encode(texto)
            return {"audio": waveform, "texto": texto, "entrada_ids": encoder.ids}

        return {"audio": waveform, "texto": texto}

def obter_dataset(
    batch_size=32,
    num_exemplos=None,
    num_workers=4,
):
    raw_dataset = datasets.load_dataset(
        "m-aliabbas/idrak_timit_subsample1",
        split="train"
    )

    tokenizadordron = tokenizador()
    dataset_processado = DatasetAudio(
        raw_dataset,
        tokenizador=tokenizadordron,
        num_examplos=num_exemplos,
    )

    dataloader = DataLoader(
        dataset_processado,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers
    )

    return dataloader
