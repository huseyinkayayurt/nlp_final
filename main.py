import torch
from transformers import AutoTokenizer

from model_test import retrival_success
from model_train import fine_tune_model
from model_utils import load_model_and_tokenizer
from train.data_loader import process_dataset
from train.model_utils import CustomSequenceClassificationModel
from train.training_pipeline import train_model


def main():
    retrival_success()

    base_model = "jinaai/jina-embeddings-v3"

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    processed_data = process_dataset("data_set/train.csv", tokenizer)
    train_data, eval_data = processed_data[:20], processed_data[20:100]
    model = CustomSequenceClassificationModel(base_model_name=base_model, num_labels=5)
    train_model(model, tokenizer, train_data, eval_data)
    print("Eğitim tamamlandı. Model kaydedildi.")

    save_directory = "fine_tuned_model"
    loaded_model, loaded_tokenizer = load_model_and_tokenizer(save_directory)


if __name__ == "__main__":
    main()
