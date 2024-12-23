from transformers import AutoTokenizer

from retrival.chunks import collect_chunks_with_indices
from retrival.data_loader import load_data
from retrival.evaluate import evaluate_top_k_accuracy
from retrival.test import retrival_success
from model_utils import load_model_and_tokenizer, load_model_and_tokenizer_trained
from retrival.visualize import plot_top_k_accuracies
from train.data_loader import process_dataset
from train.model import CustomSequenceClassificationModel
from train.training_pipeline import train_model
from trained_retrival.embedding_utils import create_embeddings, save_embeddings, load_embeddings


def main():
    data_set_file_path = "data_set/train.csv"
    pre_trained_retrival_output_folder_name = "pre_trained_retrival_output/"
    trained_retrival_output_folder_name = "trained_retrival_output/"
    trained_model_saved_directory = "fine_tuned_model"
    chunk_embeddings_path = f"{trained_retrival_output_folder_name}chunks_embeddings.pt"
    question_embeddings_path = f"{trained_retrival_output_folder_name}question_embeddings.pt"
    top_k_accuracies_output = f"{trained_retrival_output_folder_name}top_k_accuracies.png"

    model_name = "jinaai/jina-embeddings-v3"

    # tokenizer, model = load_model_and_tokenizer(model_name)
    #
    # retrival_success(data_set_file_path, pre_trained_retrival_output_folder_name, model, tokenizer)
    #
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = CustomSequenceClassificationModel(base_model_name=model_name, num_labels=5)
    #
    # processed_data = process_dataset(data_set_file_path, tokenizer)
    # train_data, eval_data = processed_data[:20], processed_data[20:100]
    #
    # train_model(model, tokenizer, train_data, eval_data, trained_model_saved_directory)
    # print("Eğitim tamamlandı. Model kaydedildi.")

    loaded_model, loaded_tokenizer = load_model_and_tokenizer_trained(trained_model_saved_directory)

    data = load_data(data_set_file_path, size=1000)
    all_chunks, correct_indices = collect_chunks_with_indices(data)

    # Chunk embedding oluşturma
    print("Chunk embedding'ler oluşturuluyor...")
    chunks_embeddings = create_embeddings(all_chunks, loaded_model, loaded_tokenizer)
    save_embeddings(chunks_embeddings, chunk_embeddings_path)

    # Soru embedding oluşturma
    print("Soru embedding'ler oluşturuluyor...")
    questions = [item["question"] for item in data]
    questions_embeddings = create_embeddings(questions, loaded_model, loaded_tokenizer)
    save_embeddings(questions_embeddings, question_embeddings_path)

    # Embedding'leri yükleme
    print("Embedding'ler yükleniyor...")
    chunks_embeddings = load_embeddings(chunk_embeddings_path)
    questions_embeddings = load_embeddings(question_embeddings_path)

    # Top-K doğruluk hesaplama
    print("Top-1 ve Top-5 doğruluk hesaplanıyor...")
    # top1_accuracy, top5_accuracy = evaluate_top_k_accuracy(
    #     questions_embeddings, chunks_embeddings, correct_indices, k=5
    # )
    top_k_accuracies = evaluate_top_k_accuracy(questions_embeddings, chunks_embeddings, correct_indices, [1, 5])
    print("Başarı oranları:", top_k_accuracies)

    plot_top_k_accuracies(top_k_accuracies["top_1"], top_k_accuracies["top_5"], top_k_accuracies_output)


if __name__ == "__main__":
    main()
