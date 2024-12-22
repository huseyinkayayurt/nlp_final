import os
import torch

from chunks import collect_chunks_with_indices
from data_loader import load_data
from embeddings import calculate_and_save_embeddings, calculate_question_embeddings
from evaluate import evaluate_top_k_accuracy
from model import load_model_and_tokenizer
from visualize import visualize_embeddings, visualize_top_k_accuracies


def main():
    """
    Ana fonksiyon, veri setini yükler ve bir örneği konsola yazdırır.
    """
    file_path = "data_set/train.csv"
    data = load_data(file_path, size=1000)

    output_folder_name = "output/"
    folder_name = os.path.dirname(output_folder_name)
    os.makedirs(folder_name, exist_ok=True)

    if data:
        all_chunks, correct_indices = collect_chunks_with_indices(data)

        model_name = "jinaai/jina-embeddings-v3"
        embeddings_file = f"{output_folder_name}_chunks_embeddings.pt"

        tokenizer, model = load_model_and_tokenizer(model_name)
        # calculate_and_save_embeddings(all_chunks, model, tokenizer, embeddings_file)

        embeddings_tsne_output = f"{output_folder_name}_chunks_embeddings_tsne.png"
        visualize_embeddings(embeddings_file, embeddings_tsne_output)

        # Embedding dosyalarını yükle
        question_embeddings = calculate_question_embeddings(data, model, tokenizer)
        chunk_embeddings = torch.load(embeddings_file)

        # Top-1 ve Top-5 başarı oranlarını hesapla
        print("Top-k başarı oranları hesaplanıyor...")
        top_k_accuracies = evaluate_top_k_accuracy(
            question_embeddings=question_embeddings,
            chunk_embeddings=chunk_embeddings,
            correct_indices=correct_indices,
            top_k_values=[1, 5]
        )
        print("Başarı oranları:", top_k_accuracies)

        # Başarı oranlarını görselleştir
        top_k_accuracies_output = f"{output_folder_name}_top_k_accuracies.png"
        visualize_top_k_accuracies(top_k_accuracies, top_k_accuracies_output)


    else:
        print("Veri kümesi yüklenemedi.")


if __name__ == "__main__":
    main()
