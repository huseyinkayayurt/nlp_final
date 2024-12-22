from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm


def evaluate_top_k_accuracy(question_embeddings, chunk_embeddings, correct_indices, top_k_values=None):
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
