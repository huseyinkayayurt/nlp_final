from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm


def evaluate_top_k_accuracy(question_embeddings, chunk_embeddings, correct_indices, top_k_values=None):
    """
    Top-k doğruluk oranlarını hesaplar.

    Args:
        question_embeddings (np.ndarray): Soruların embedding'leri.
        chunk_embeddings (np.ndarray): Chunk'ların embedding'leri.
        correct_indices (dict): Her bir sorunun doğru chunk indeksini belirten sözlük.
        top_k_values (list): Hangi k değerleri için doğruluk hesaplanacağı.

    Returns:
        dict: Top-k doğruluk oranları.
    """
    if top_k_values is None:
        top_k_values = [1, 5]
    top_k_accuracies = {f"top_{k}": 0 for k in top_k_values}
    num_questions = len(question_embeddings)

    print("Top-k doğruluk oranları hesaplanıyor...")
    for question_idx, question_embedding in tqdm(enumerate(question_embeddings), total=num_questions, desc="Top-k hesaplama"):
        # Sorunun doğru chunk indeksini al
        correct_chunk_idx = correct_indices[question_idx]

        # Tüm chunk'larla benzerlik hesapla
        similarities = cosine_similarity([question_embedding], chunk_embeddings)[0]

        # Benzerliklere göre sıralama
        ranked_indices = np.argsort(similarities)[::-1]

        # Top-k doğruluk kontrolü
        for k in top_k_values:
            if correct_chunk_idx in ranked_indices[:k]:
                top_k_accuracies[f"top_{k}"] += 1

    # Oranları normalize et
    for k in top_k_values:
        top_k_accuracies[f"top_{k}"] /= num_questions

    return top_k_accuracies

# def evaluate_top_k_accuracy(question_embeddings, chunk_embeddings, correct_indices, top_k_list=[1, 5]):
#     """
#     Top-k başarı oranlarını hesaplar.
#
#     Args:
#         question_embeddings (np.ndarray): Soruların embedding'leri (N x D).
#         chunk_embeddings (np.ndarray): Chunk embedding'leri (M x D).
#         correct_indices (List[int]): Her soruya karşılık gelen doğru chunk indeksleri.
#         top_k_list (List[int]): Hesaplanacak top-k başarı değerleri.
#
#     Returns:
#         dict: Top-k başarı oranları.
#     """
#     # Cosine Similarity hesaplama
#     print("Cosine Similarity hesaplanıyor...")
#     similarities = cosine_similarity(question_embeddings, chunk_embeddings)
#
#     top_k_accuracies = {f"top-{k}": 0 for k in top_k_list}
#     num_questions = len(question_embeddings)
#
#     # Her soru için doğru chunk'ın top-k içinde olup olmadığını kontrol et
#     for i, correct_index in enumerate(correct_indices):
#         ranked_indices = np.argsort(similarities[i])[::-1]  # Benzerliğe göre sırala (büyükten küçüğe)
#         for k in top_k_list:
#             if correct_index in ranked_indices[:k]:
#                 top_k_accuracies[f"top-{k}"] += 1
#
#     # Oranları hesapla
#     for k in top_k_list:
#         top_k_accuracies[f"top-{k}"] /= num_questions
#
#     return top_k_accuracies
