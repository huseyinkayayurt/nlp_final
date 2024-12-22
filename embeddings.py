import torch
from tqdm import tqdm
import numpy as np


def calculate_and_save_embeddings(chunks, model, tokenizer, output_file):
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


def calculate_question_embeddings(data, model, tokenizer, batch_size=16, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Soruların embedding'lerini hesaplar.

        :param device:
        :param batch_size:
        :param data:
        :param tokenizer:
        :param model:
    """

    # Soruları al
    questions = [item["question"] for item in data]

    embeddings = []
    print("Embedding soruları:")
    for i in tqdm(range(0, len(questions), batch_size), desc="Embedding hesaplanıyor"):
        batch = questions[i:i + batch_size]
        with torch.no_grad():
            inputs = tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
            outputs = model(**inputs)
            # Genelde son gizli katmanın ortalaması alınır
            batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            embeddings.append(batch_embeddings)

    embeddings = np.vstack(embeddings)
    print(f"Embedding boyutları: {embeddings.shape}")
    return embeddings
