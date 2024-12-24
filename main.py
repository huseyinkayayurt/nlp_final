from sklearn.model_selection import train_test_split
from data.loader import load_data
from data.process import process_dataset
from model.utils import load_model_and_tokenizer_pre_trained, load_model_and_tokenizer_for_train, load_model_and_tokenizer_trained
from retrival import operation_retrival
from train.pipeline import train_model
import utils.constants as Constants


def main():
    data = load_data(Constants.DATA_SET_FILE_PATH, size=1000)

    pre_trained_tokenizer, pre_trained_model = load_model_and_tokenizer_pre_trained(Constants.BASE_MODEL_NAME)
    operation_retrival(data, pre_trained_model, pre_trained_tokenizer, Constants.PRE_TRAINED_CHUNK_EMBEDDINGS_PATH,
                       Constants.PRE_TRAINED_QUESTION_EMBEDDINGS_PATH,
                       Constants.PRE_TRAINED_CHUNK_EMBEDDINGS_TSNE_PATH, Constants.PRE_TRAINED_QUESTION_EMBEDDINGS_TSNE_PATH,
                       Constants.PRE_TRAINED_COMBINE_EMBEDDINGS_TSNE_PATH,
                       Constants.PRE_TRAINED_TOP_K_ACCURACIES_OUTPUT)

    for_train_tokenizer, for_train_model = load_model_and_tokenizer_for_train(Constants.BASE_MODEL_NAME)
    processed_data = process_dataset(Constants.DATA_SET_FILE_PATH, for_train_tokenizer)
    train_dataset, eval_dataset = train_test_split(processed_data, test_size=0.2, random_state=42)
    train_model(for_train_model, for_train_tokenizer, train_dataset, eval_dataset, Constants.TRAINED_MODEL_DIRECTORY)

    trained_tokenizer, trained_model = load_model_and_tokenizer_trained(Constants.TRAINED_MODEL_DIRECTORY)
    operation_retrival(data, trained_model.base_model, trained_tokenizer, Constants.TRAINED_CHUNK_EMBEDDINGS_PATH,
                       Constants.TRAINED_QUESTION_EMBEDDINGS_PATH,
                       Constants.TRAINED_CHUNK_EMBEDDINGS_TSNE_PATH, Constants.TRAINED_QUESTION_EMBEDDINGS_TSNE_PATH,
                       Constants.TRAINED_COMBINE_EMBEDDINGS_TSNE_PATH,
                       Constants.TRAINED_TOP_K_ACCURACIES_OUTPUT)

    # operation_retrival(data, pre_trained_retrival_output_folder_name, model, tokenizer)
    #
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = CustomSequenceClassificationModel(base_model_name=model_name, num_labels=5)
    #
    # processed_data = process_dataset(data_set_file_path, tokenizer)
    # train_data, eval_data = processed_data[:20], processed_data[20:100]
    #
    # train_model(model, tokenizer, train_data, eval_data, trained_model_saved_directory)
    # print("Eğitim tamamlandı. Model kaydedildi.")
    #
    # loaded_model, loaded_tokenizer = load_model_and_tokenizer_trained(trained_model_saved_directory)
    #
    # data = load_data(data_set_file_path, size=1000)
    # all_chunks, correct_indices = collect_chunks_with_indices(data)
    #
    # # Chunk embedding oluşturma
    # print("Chunk embedding'ler oluşturuluyor...")
    # chunks_embeddings = create_embeddings(all_chunks, loaded_model, loaded_tokenizer)
    # save_embeddings(chunks_embeddings, chunk_embeddings_path)
    #
    # # Soru embedding oluşturma
    # print("Soru embedding'ler oluşturuluyor...")
    # questions = [item["question"] for item in data]
    # questions_embeddings = create_embeddings(questions, loaded_model, loaded_tokenizer)
    # save_embeddings(questions_embeddings, question_embeddings_path)
    #
    # # Embedding'leri yükleme
    # print("Embedding'ler yükleniyor...")
    # chunks_embeddings = load_embeddings(chunk_embeddings_path)
    # questions_embeddings = load_embeddings(question_embeddings_path)
    #
    # # Top-K doğruluk hesaplama
    # print("Top-1 ve Top-5 doğruluk hesaplanıyor...")
    # # top1_accuracy, top5_accuracy = evaluate_top_k_accuracy(
    # #     questions_embeddings, chunks_embeddings, correct_indices, k=5
    # # )
    # top_k_accuracies = evaluate_top_k_accuracy(questions_embeddings, chunks_embeddings, correct_indices, [1, 5])
    # print("Başarı oranları:", top_k_accuracies)
    #
    # plot_top_k_accuracies(top_k_accuracies["top_1"], top_k_accuracies["top_5"], top_k_accuracies_output)


if __name__ == "__main__":
    main()
