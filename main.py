from model_test import retrival_success
from model_train import fine_tune_model


def main():
    # retrival_success()
    base_model = "jinaai/jina-embeddings-v3"
    dataset_path = "data_set/train.csv"
    output_dir = "fine_tuned_model"

    # fine_tune_model(base_model, dataset_path, output_dir)


if __name__ == "__main__":
    main()
