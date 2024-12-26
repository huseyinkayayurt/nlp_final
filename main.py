from sklearn.model_selection import train_test_split
from data.loader import load_data
from data.process import process_dataset
from model.utils import load_model_and_tokenizer_pre_trained, load_model_and_tokenizer_for_train, load_model_and_tokenizer_trained
from retrival import operation_retrival
from train.pipeline import train_model
import utils.constants as Constants


def main():
    data = load_data(Constants.DATA_SET_FILE_PATH, size=10)

    pre_trained_tokenizer, pre_trained_model = load_model_and_tokenizer_pre_trained(Constants.BASE_MODEL_NAME)
    pre_trained_chunks_embeddings, pre_trained_questions_embeddings = operation_retrival(data,
                                                                                         pre_trained_model,
                                                                                         pre_trained_tokenizer,
                                                                                         Constants.PRE_TRAINED_CHUNK_EMBEDDINGS_PATH,
                                                                                         Constants.PRE_TRAINED_QUESTION_EMBEDDINGS_PATH,
                                                                                         Constants.PRE_TRAINED_CHUNK_EMBEDDINGS_TSNE_PATH,
                                                                                         Constants.PRE_TRAINED_QUESTION_EMBEDDINGS_TSNE_PATH,
                                                                                         Constants.PRE_TRAINED_COMBINE_EMBEDDINGS_TSNE_PATH,
                                                                                         Constants.PRE_TRAINED_TOP_K_ACCURACIES_OUTPUT)

    for_train_tokenizer, for_train_model = load_model_and_tokenizer_for_train(Constants.BASE_MODEL_NAME)
    processed_data = process_dataset(Constants.DATA_SET_FILE_PATH, for_train_tokenizer, data)
    train_dataset, eval_dataset = train_test_split(processed_data, test_size=0.2, random_state=42)
    train_model(for_train_model, for_train_tokenizer, train_dataset, eval_dataset, Constants.TRAINED_MODEL_DIRECTORY)

    trained_tokenizer, trained_model = load_model_and_tokenizer_trained(Constants.TRAINED_MODEL_DIRECTORY)
    trained_chunks_embeddings, trained_questions_embeddings = operation_retrival(data,
                                                                                 trained_model,
                                                                                 trained_tokenizer,
                                                                                 Constants.TRAINED_CHUNK_EMBEDDINGS_PATH,
                                                                                 Constants.TRAINED_QUESTION_EMBEDDINGS_PATH,
                                                                                 Constants.TRAINED_CHUNK_EMBEDDINGS_TSNE_PATH,
                                                                                 Constants.TRAINED_QUESTION_EMBEDDINGS_TSNE_PATH,
                                                                                 Constants.TRAINED_COMBINE_EMBEDDINGS_TSNE_PATH,
                                                                                 Constants.TRAINED_TOP_K_ACCURACIES_OUTPUT)


if __name__ == "__main__":
    main()
