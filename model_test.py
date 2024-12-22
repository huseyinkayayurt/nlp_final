import os

from chunks import collect_chunks_with_indices
from data_loader import load_data
from embeddings import load_embeddings
from evaluate import evaluate_top_k_accuracy
from model import load_model_and_tokenizer
from visualize import visualize_embeddings, visualize_embeddings_combined, plot_top_k_accuracies


def retrival_success():
    file_path = "data_set/train.csv"
    data = load_data(file_path, size=1000)

    output_folder_name = "output/"
    folder_name = os.path.dirname(output_folder_name)
    os.makedirs(folder_name, exist_ok=True)

    if data:
        all_chunks, correct_indices = collect_chunks_with_indices(data)

        model_name = "jinaai/jina-embeddings-v3"
        chunks_embeddings_file = f"{output_folder_name}_chunks_embeddings.pt"
        questions_embeddings_file = f"{output_folder_name}_questions_embeddings.pt"
        embeddings_combine_tsne_output = f"{output_folder_name}_embeddings_combine_tsne.png"
        embeddings_chunk_tsne_output = f"{output_folder_name}_embeddings_chunk_tsne.png"
        embeddings_question_tsne_output = f"{output_folder_name}_embeddings_question_tsne.png"
        top_k_accuracies_output = f"{output_folder_name}_top_k_accuracies.png"

        tokenizer, model = load_model_and_tokenizer(model_name)

        # chunks_embeddings = calculate_chunk_embeddings(all_chunks, model, tokenizer)
        # save_embeddings(chunks_embeddings, chunks_embeddings_file)

        # questions_embeddings = calculate_question_embeddings(data, model, tokenizer)
        # save_embeddings(questions_embeddings, questions_embeddings_file)

        chunk_embeddings = load_embeddings(chunks_embeddings_file)
        question_embeddings = load_embeddings(questions_embeddings_file)

        visualize_embeddings(chunk_embeddings, embeddings_chunk_tsne_output)
        visualize_embeddings(question_embeddings, embeddings_question_tsne_output)
        visualize_embeddings_combined(chunk_embeddings, question_embeddings, embeddings_combine_tsne_output)

        top_k_accuracies = evaluate_top_k_accuracy(question_embeddings, chunk_embeddings, correct_indices, [1, 5])
        print("Başarı oranları:", top_k_accuracies)

        plot_top_k_accuracies(top_k_accuracies["top_1"], top_k_accuracies["top_5"], top_k_accuracies_output)


    else:
        print("Veri kümesi yüklenemedi.")
