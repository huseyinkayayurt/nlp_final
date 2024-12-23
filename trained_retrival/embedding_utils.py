import torch
from tqdm import tqdm


def create_embeddings(data, model, tokenizer, max_length=512, device="cpu"):
    """
    Veri için embedding oluşturur.

    Args:
        data (list): Veri kümesi (chunk veya soru).
        model (nn.Module): Embedding modeli.
        tokenizer (AutoTokenizer): Tokenizer.
        max_length (int): Maksimum sekans uzunluğu.
        device (str): Cihaz ("cpu" veya "cuda").

    Returns:
        list: Embedding'ler.
    """
    embeddings = []

    model.eval()
    model.to(device)

    for item in tqdm(data, desc="Embedding oluşturuluyor"):
        inputs = tokenizer(
            item,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model.base_model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            embeddings.append(embedding)

    return embeddings


def save_embeddings(embeddings, output_path):
    """
    Embedding'leri bir dosyaya kaydeder.

    Args:
        embeddings (list): Embedding'ler.
        output_path (str): Kaydedilecek dosya yolu.
    """
    torch.save(embeddings, output_path)


def load_embeddings(file_path):
    """
    Kaydedilmiş embedding'leri yükler.

    Args:
        file_path (str): Dosya yolu.

    Returns:
        list: Embedding'ler.
    """
    return torch.load(file_path)
