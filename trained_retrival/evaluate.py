import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def evaluate_top_k_accuracy(questions_embeddings, chunks_embeddings, correct_indices, k=5):
    """
    Top-K doğruluk oranlarını hesaplar.

    Args:
        questions_embeddings (list): Soruların embedding'leri.
        chunks_embeddings (list): Chunk'ların embedding'leri.
        correct_indices (dict): Doğru chunk indeksleri.
        k (int): K değeri.

    Returns:
        float: Top-1 doğruluk.
        float: Top-K doğruluk.
    """
    correct_top1 = 0
    correct_topk = 0

    similarities = cosine_similarity(questions_embeddings, chunks_embeddings)
    for idx, similarity in enumerate(similarities):
        correct_chunk = correct_indices[idx]
        top_k_indices = np.argsort(similarity)[-k:]  # En yüksek K benzerlik

        if correct_chunk in top_k_indices:
            correct_topk += 1

        if correct_chunk == np.argmax(similarity):  # En yüksek benzerlik
            correct_top1 += 1

    total = len(questions_embeddings)
    return correct_top1 / total, correct_topk / total
