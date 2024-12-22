import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def visualize_embeddings(embeddings_file, output_image):
    """
    TSNE kullanarak embedding görselleştir ve kaydet.

    Args:
        embeddings_file (str): Embedding'lerin kaydedildiği dosya.
        output_image (str): Görselleştirmenin kaydedileceği dosya adı.
    """
    # Embedding'leri yükle
    embeddings = torch.load(embeddings_file)

    print("TSNE uygulanıyor...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Görselleştirme
    plt.figure(figsize=(10, 8))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=10, alpha=0.6)
    plt.title("Chunk Embedding'lerinin TSNE Görselleştirmesi")
    plt.savefig(output_image)
    plt.close()
    print(f"TSNE görseli kaydedildi: {output_image}")

def visualize_top_k_accuracies(top_k_accuracies, output_image):
    """
    Top-k başarı oranlarını bir grafik olarak görselleştirir ve kaydeder.

    Args:
        top_k_accuracies (dict): Top-k başarı oranları.
        output_image (str): Görselin kaydedileceği dosya adı.
    """
    labels = list(top_k_accuracies.keys())
    values = list(top_k_accuracies.values())

    plt.figure(figsize=(8, 6))
    plt.bar(labels, values, color=['blue', 'green'], alpha=0.7)
    plt.title("Top-k Başarı Oranları")
    plt.xlabel("Top-k")
    plt.ylabel("Başarı Oranı")
    plt.ylim(0, 1)
    plt.savefig(output_image)
    plt.close()
    print(f"Başarı oranları grafiği kaydedildi: {output_image}")
