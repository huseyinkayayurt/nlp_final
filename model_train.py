import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict
import torch


def prepare_fine_tune_dataset(file_path):
    """
    Eğitim için veri kümesini hazırlar.

    Args:
        file_path (str): Veri kümesinin CSV dosyasının yolu.

    Returns:
        DatasetDict: Eğitim ve test veri kümeleri.
    """
    import pandas as pd

    # CSV'yi yükle
    df = pd.read_csv(file_path)

    # Veri formatı: `input` (question + context), `label` (correct chunk index)
    df["input"] = df["question"] + " " + df["context"]
    df = df[["input", "correct_intro_idx"]]

    # Hugging Face Dataset formatına dönüştürme
    dataset = Dataset.from_pandas(df)

    # Veri kümesini eğitim ve test olarak ayırma
    dataset = dataset.train_test_split(test_size=0.2)
    return DatasetDict({"train": dataset["train"], "test": dataset["test"]})


def fine_tune_model(base_model_name, dataset_path, output_dir, epochs=3, batch_size=8):
    # Model ve tokenizer yükle
    print(f"Model yükleniyor: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(base_model_name, num_labels=5)  # 5 chunk için

    # Dataset hazırla
    dataset = prepare_fine_tune_dataset(dataset_path)

    # Tokenizer ile veriyi işleme
    def preprocess_function(examples):
        return tokenizer(examples["input"], truncation=True, padding="max_length", max_length=512)

    tokenized_datasets = dataset.map(preprocess_function, batched=True)

    # Kullanılmayan sütunları kaldır
    tokenized_datasets = tokenized_datasets.remove_columns(["input"])
    tokenized_datasets = tokenized_datasets.rename_column("correct_intro_idx", "labels")
    tokenized_datasets.set_format("torch")

    # Eğitim ayarları
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        save_total_limit=1
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
    )

    # Eğitim
    print("Eğitim başlıyor...")
    trainer.train()

    # Modeli kaydet
    print(f"Eğitim tamamlandı. Model kaydediliyor: {output_dir}")
    trainer.save_model(output_dir)
