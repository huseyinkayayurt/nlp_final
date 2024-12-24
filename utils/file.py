import torch
import numpy as np


def save_embeddings(embeddings, output_file):
    # Embedding'leri kaydet
    embeddings = np.array(embeddings)
    torch.save(embeddings, output_file)
    print(f"Embedding'ler kaydedildi: {output_file}")


def load_embeddings(file):
    embeddings = torch.load(file, weights_only=False)
    return embeddings
