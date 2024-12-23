import csv
import os

import torch
from sklearn.model_selection import train_test_split
from transformers import TrainingArguments, Trainer, AutoModelForSequenceClassification, AutoTokenizer, AutoModel
from torch.utils.data import Dataset
from torch import nn
from model import load_model_and_tokenizer


class CustomSequenceClassificationModel(nn.Module):
    def __init__(self, base_model_name, num_labels):
        super(CustomSequenceClassificationModel, self).__init__()
        self.num_labels = num_labels

        # Temel modeli yükle
        self.base_model = AutoModel.from_pretrained(base_model_name, trust_remote_code=True, ignore_mismatched_sizes=True)
        hidden_size = self.base_model.config.hidden_size

        # Sınıflandırma için bir head ekleyin
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        # Temel modelden çıktılar alın
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # CLS token

        # Sınıflandırma katmanına gönderin
        logits = self.classifier(pooled_output)

        # Eğer label varsa, kaybı hesapla
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return {"loss": loss, "logits": logits}


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "input_ids": self.data[idx]["input_ids"],
            "attention_mask": self.data[idx]["attention_mask"],
            "labels": torch.tensor(self.data[idx]["labels"], dtype=torch.long),
        }


def process_dataset(file_path):
    """
    Veri setini işler, context parçalarını chunklara ayırır ve doğru cevabın indeksini belirler.
    Args:
        file_path (str): Veri setinin CSV dosya yolu.
    Returns:
        list: İşlenmiş veri kümesi [{question, chunks, label}, ...]
    """
    processed_data = []
    with open(file_path, "r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Context'i chunk'lara ayır
            context = row["context"]
            split_points = eval(row["ctx_split_points"])
            chunks = [
                context[start:end].strip()
                for start, end in zip([0] + split_points, split_points + [len(context)])
            ]

            # Veri yapısını oluştur
            processed_data.append({
                "question": row["question"],
                "chunks": chunks,
                "label": int(row["correct_intro_idx"])  # Doğru chunk indeksi
            })

    return processed_data


def preprocess_data(data, tokenizer, max_length=512):
    processed_data = []
    for example in data:
        question = example["question"]
        context = " ".join(example["chunks"])  # Chunk'ları birleştir
        inputs = tokenizer(
            question, context,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        processed_data.append({
            "input_ids": inputs["input_ids"].squeeze(0),  # Tensor'u düzleştir
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": int(example["label"]),  # Doğru cevabın indexi
        })
    return processed_data


def fine_tune_model(model_name):
    # tokenizer, model = load_model_and_tokenizer(model_name)
    num_labels = 5
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = CustomSequenceClassificationModel(base_model_name=model_name, num_labels=num_labels)
    raw_dataset = process_dataset("data_set/train.csv")
    raw_dataset = raw_dataset[:100]
    train_data, eval_data = train_test_split(raw_dataset, test_size=0.2, random_state=42)

    train_dataset = preprocess_data(train_data, tokenizer)
    eval_dataset = preprocess_data(eval_data, tokenizer)

    train_dataset = CustomDataset(train_dataset)
    eval_dataset = CustomDataset(eval_dataset)

    print("Eğitim veri setindeki ilk eleman:", train_dataset[0])
    example = train_dataset[0]

    # Model girdileri
    inputs = {
        "input_ids": example["input_ids"].unsqueeze(0),
        "attention_mask": example["attention_mask"].unsqueeze(0),
        "labels": example["labels"].unsqueeze(0),
    }

    # Modeli çalıştır
    outputs = model(**inputs)

    # Çıktıları kontrol edin
    print(outputs.keys())  # "loss" anahtarını içerdiğinden emin olun

    training_args = TrainingArguments(
        output_dir="trained_model",
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
        train_dataset=train_dataset,  # Tokenize edilmiş veri kümesi
        eval_dataset=eval_dataset,  # Tokenize edilmiş veri kümesi
        tokenizer=tokenizer,
    )

    trainer.train()

    output_model_folder_name = "fine_tuned_model/"
    folder_name = os.path.dirname(output_model_folder_name)
    os.makedirs(folder_name, exist_ok=True)
    
    torch.save(model.state_dict(), f"{output_model_folder_name}pytorch_model.bin")
    model_config = {
        "model_name": "CustomSequenceClassificationModel",
        "base_model_name": model.base_model.config._name_or_path,
        "num_labels": model.num_labels,
    }
    with open("fine_tuned_model/config.json", "w") as f:
        import json
        json.dump(model_config, f)

    tokenizer.save_pretrained("fine_tuned_model")

    # model.save_pretrained("fine_tuned_model")
    # tokenizer.save_pretrained("fine_tuned_model")
