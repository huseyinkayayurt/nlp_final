import torch
from tqdm import tqdm
import numpy as np

def calculate_and_save_embeddings(chunks, model,tokenizer, output_file):
    """
    Chunk'lar için embedding hesapla ve kaydet.

    Args:
        chunks (List[str]): Chunk metinleri.
        output_file (str): Embedding'lerin kaydedileceği dosya adı.
        :param output_file:
        :param chunks:
        :param tokenizer:
        :param model:
    """

    embeddings = []

    # Embedding hesaplama
    print("Embedding hesaplanıyor...")
    with torch.no_grad():
        for chunk in tqdm(chunks, desc="Embedding chunk'lar"):
            inputs = tokenizer(chunk, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
            outputs = model(**inputs)
            chunk_embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0).numpy()
            embeddings.append(chunk_embedding)

    # Embedding'leri kaydet
    embeddings = np.array(embeddings)
    torch.save(embeddings, output_file)
    print(f"Embedding'ler kaydedildi: {output_file}")
