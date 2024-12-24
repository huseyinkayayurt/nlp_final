import os
from transformers import Trainer, TrainingArguments
from torch.utils.data import Dataset

from model.utils import save_model_and_tokenizer


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def train_model(model, tokenizer, train_data, eval_data, output_dir):
    """
    Modeli eğitir ve çıktıyı kaydeder.

    Args:
        model (nn.Module): Eğitim yapılacak model.
        tokenizer (AutoTokenizer): Tokenizer.
        train_data (list): Eğitim verisi.
        eval_data (list): Değerlendirme verisi.
        output_dir (str): Modelin kaydedileceği dizin.
    """
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="logs",
        push_to_hub=False,
        remove_unused_columns=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=CustomDataset(train_data),
        eval_dataset=CustomDataset(eval_data),
        tokenizer=tokenizer,
    )

    trainer.train()

    save_model_and_tokenizer(model, tokenizer, output_dir)
    print("Eğitim tamamlandı. Model kaydedildi.")
