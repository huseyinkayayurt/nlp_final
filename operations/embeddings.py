import torch
from tqdm import tqdm
import numpy as np


def calculate_chunk_embeddings(chunks, model, tokenizer):
    embeddings = []

    # Embedding hesaplama
    print("Embedding hesaplanıyor...")
    with torch.no_grad():
        for chunk in tqdm(chunks, desc="Embedding chunk'lar"):
            inputs = tokenizer(chunk, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
            outputs = model(**inputs)
            chunk_embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0).numpy()
            embeddings.append(chunk_embedding)

    return embeddings


def calculate_question_embeddings(data, model, tokenizer, batch_size=16, device="cuda" if torch.cuda.is_available() else "cpu"):
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
