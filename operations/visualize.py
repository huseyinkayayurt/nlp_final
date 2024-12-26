import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np


def visualize_embeddings_combined(chunk_embeddings, question_embeddings, output_path):
    # t-SNE ile boyut indirgeme
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=3000)
    all_embeddings = np.vstack([chunk_embeddings, question_embeddings])
    tsne_results = tsne.fit_transform(all_embeddings)

    # Embedding'leri böl
    num_chunks = chunk_embeddings.shape[0]
    chunk_tsne = tsne_results[:num_chunks]
    question_tsne = tsne_results[num_chunks:]

    # Görselleştirme
    plt.figure(figsize=(10, 8))
    plt.scatter(chunk_tsne[:, 0], chunk_tsne[:, 1], c="blue", label="Chunks", alpha=0.6, s=20)
    plt.scatter(question_tsne[:, 0], question_tsne[:, 1], c="red", label="Questions", alpha=0.8, s=50)

    plt.title("t-SNE Visualization of Embeddings", fontsize=16)
    plt.xlabel("Dimension 1", fontsize=12)
    plt.ylabel("Dimension 2", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.4)

    # Çıktıyı kaydet
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"TSNE görseli kaydedildi: {output_path}")


def visualize_embeddings(embeddings, output_image, perplexity=30):
    print("TSNE uygulanıyor...")
    n_samples = embeddings.shape[0]
    if perplexity >= n_samples:
        perplexity = max(5, n_samples // 2)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Görselleştirme
    plt.figure(figsize=(10, 8))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=10, alpha=0.6)
    plt.title("Chunk Embedding'lerinin TSNE Görselleştirmesi")
    plt.savefig(output_image)
    plt.close()
    print(f"TSNE görseli kaydedildi: {output_image}")


def plot_top_k_accuracies(top1_acc, top5_acc, output_path):
    """
    Top-1 ve Top-5 doğruluk oranlarını bir bar grafiği ile görselleştirir.

    Args:
        top1_acc (float): Top-1 doğruluk oranı.
        top5_acc (float): Top-5 doğruluk oranı.
        output_path (str): Çıktı grafiğinin kaydedileceği dosya yolu.
    """
    # Bar grafiği verileri
    labels = ['Top-1', 'Top-5']
    values = [top1_acc, top5_acc]
    colors = ['blue', 'green']

    # Bar grafiği oluşturma
    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, values, color=colors, alpha=0.7)
    plt.ylim(0, 1.05)  # Y eksenini 1.05'e kadar ayarla (üst sınır biraz daha yukarıda olsun)

    # Barların üzerinde değerleri göster
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 0.05,  # Yerden biraz aşağıda
                 f"{value:.3f}",  # Ondalıklı format (3 basamak)
                 ha='center', va='bottom', fontsize=12, color='black')

    # Başlık ve etiketler
    plt.title("Top-k Accuracy", fontsize=16)
    plt.ylabel("Accuracy", fontsize=14)
    plt.xlabel("Metrics", fontsize=14)
    plt.grid(axis='y', alpha=0.4)

    # Grafiği kaydet ve göster
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
