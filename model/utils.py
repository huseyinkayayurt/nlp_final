import os
import json
import torch
from transformers import AutoTokenizer, AutoModel
from train.model import CustomSequenceClassificationModel
import einops  # jinaai/jina-embeddings-v3 dil modeli için gerekli

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def save_model_and_tokenizer(model, tokenizer, save_directory):
    os.makedirs(save_directory, exist_ok=True)

    # Modeli kaydet
    torch.save(model.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))

    # Model konfigürasyonunu kaydet
    config = {
        "model_type": "CustomSequenceClassificationModel",
        "base_model_name": model.base_model.config._name_or_path,
        "num_labels": model.num_labels,
    }
    with open(os.path.join(save_directory, "config.json"), "w") as f:
        json.dump(config, f)

    # Tokenizer'ı kaydet
    tokenizer.save_pretrained(save_directory)


def load_model_and_tokenizer_trained(save_directory, device="cpu"):
    # Tokenizer'ı yükle
    tokenizer = AutoTokenizer.from_pretrained(save_directory)

    # Model konfigürasyonunu yükle
    with open(os.path.join(save_directory, "config.json"), "r") as f:
        config = json.load(f)

    # Modeli oluştur ve yükle
    model = CustomSequenceClassificationModel(
        base_model_name=config["base_model_name"],
        num_labels=config["num_labels"],
    )
    model.load_state_dict(torch.load(os.path.join(save_directory, "pytorch_model.bin"), map_location=device))
    model.to(device)
    model.eval()

    return tokenizer, model


def load_model_and_tokenizer_pre_trained(model_name):
    """Model ve tokenizer yükler."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True, ignore_mismatched_sizes=True)
    model.eval()

    return tokenizer, model


def load_model_and_tokenizer_for_train(model_name):
    """Model ve tokenizer yükler."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = CustomSequenceClassificationModel(base_model_name=model_name, num_labels=5)

    return tokenizer, model
