import os
import einops  # jinaai/jina-embeddings-v3 dil modeli için gerekli
from transformers import AutoTokenizer, AutoModel

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_model_and_tokenizer(model_name):
    """Model ve tokenizer yükler."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True, ignore_mismatched_sizes=True)
    model.eval()

    return tokenizer, model