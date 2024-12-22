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
