from data.chunks import collect_chunks_with_indices

from operations.embeddings import calculate_chunk_embeddings, calculate_question_embeddings, calculate_chunk_embeddings_trained
from operations.evaluate import evaluate_top_k_accuracy
from operations.visualize import visualize_embeddings, visualize_embeddings_combined, plot_top_k_accuracies
from utils.file import save_embeddings, load_embeddings


def operation_retrival(
        data,
        model,
        tokenizer,
        chunks_embeddings_file,
        questions_embeddings_file,
        embeddings_chunk_tsne_output,
        embeddings_question_tsne_output,
        embeddings_combine_tsne_output,
        top_k_accuracies_output):
    if data:
        all_chunks, correct_indices = collect_chunks_with_indices(data)

        chunks_embeddings = calculate_chunk_embeddings_trained(all_chunks, model, tokenizer)
        save_embeddings(chunks_embeddings, chunks_embeddings_file)

        questions = [item["question"] for item in data]
        questions_embeddings = calculate_chunk_embeddings_trained(questions, model, tokenizer)
        save_embeddings(questions_embeddings, questions_embeddings_file)

        chunk_embeddings = load_embeddings(chunks_embeddings_file)
        question_embeddings = load_embeddings(questions_embeddings_file)

        visualize_embeddings(chunk_embeddings, embeddings_chunk_tsne_output)
        visualize_embeddings(question_embeddings, embeddings_question_tsne_output)
        visualize_embeddings_combined(chunk_embeddings, question_embeddings, embeddings_combine_tsne_output)

        top_k_accuracies = evaluate_top_k_accuracy(question_embeddings, chunk_embeddings, correct_indices, [1, 5])
        print("Başarı oranları:", top_k_accuracies)

        plot_top_k_accuracies(top_k_accuracies["top_1"], top_k_accuracies["top_5"], top_k_accuracies_output)

        return chunk_embeddings, question_embeddings
    else:
        print("Veri kümesi yüklenemedi.")
