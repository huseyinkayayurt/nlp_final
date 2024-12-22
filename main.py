import os

from chunks import collect_chunks_with_indices
from data_loader import load_data
from embeddings import calculate_and_save_embeddings
from model import load_model_and_tokenizer
from visualize import visualize_embeddings


def main():
    """
    Ana fonksiyon, veri setini yükler ve bir örneği konsola yazdırır.
    """
    file_path = "data_set/train.csv"
    data = load_data(file_path, size=10)

    filename="output/"
    folder_name = os.path.dirname(filename)
    os.makedirs(folder_name, exist_ok=True)

    if data:
        all_chunks, correct_indices = collect_chunks_with_indices(data)

        model_name = "jinaai/jina-embeddings-v3"
        embeddings_file = f"{filename}_chunks_embeddings.pt"

        tokenizer, model = load_model_and_tokenizer(model_name)
        calculate_and_save_embeddings(all_chunks, model,tokenizer, embeddings_file)

        output_image = f"{filename}_chunks_embeddings_tsne.png"
        visualize_embeddings(embeddings_file, output_image)


    else:
        print("Veri kümesi yüklenemedi.")

if __name__ == "__main__":
    main()
