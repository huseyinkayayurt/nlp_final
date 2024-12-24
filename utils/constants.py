import os

from utils.directory import create_dir

DATA_SET_FILE_PATH = "data_set/train.csv"
BASE_MODEL_NAME = "jinaai/jina-embeddings-v3"
PRE_TRAINED_OUTPUT_FOLDER_NAME = create_dir("pre_trained_output")
TRAINED_OUTPUT_FOLDER_NAME = create_dir("trained_output")
TRAINED_MODEL_DIRECTORY = create_dir("fine_tuned_model")

CHUNKS_PATH = "chunks_embeddings.pt"
CHUNKS_TSNE_PATH = "chunks_embeddings_tsne.png"
QUESTIONS_PATH = "question_embeddings.pt"
QUESTIONS_TSNE_PATH = "questions_embeddings_tsne.png"
COMBINE_TSNE_PATH = "combine_embeddings_tsne.png"
TOP_K_ACCURACIES_PATH = "top_k_accuracies.png"

PRE_TRAINED_CHUNK_EMBEDDINGS_PATH = os.path.join(PRE_TRAINED_OUTPUT_FOLDER_NAME, CHUNKS_PATH)
PRE_TRAINED_QUESTION_EMBEDDINGS_PATH = os.path.join(PRE_TRAINED_OUTPUT_FOLDER_NAME, QUESTIONS_PATH)
PRE_TRAINED_TOP_K_ACCURACIES_OUTPUT = os.path.join(PRE_TRAINED_OUTPUT_FOLDER_NAME, TOP_K_ACCURACIES_PATH)

PRE_TRAINED_CHUNK_EMBEDDINGS_TSNE_PATH = os.path.join(PRE_TRAINED_OUTPUT_FOLDER_NAME, CHUNKS_TSNE_PATH)
PRE_TRAINED_QUESTION_EMBEDDINGS_TSNE_PATH = os.path.join(PRE_TRAINED_OUTPUT_FOLDER_NAME, QUESTIONS_TSNE_PATH)
PRE_TRAINED_COMBINE_EMBEDDINGS_TSNE_PATH = os.path.join(PRE_TRAINED_OUTPUT_FOLDER_NAME, COMBINE_TSNE_PATH)

TRAINED_CHUNK_EMBEDDINGS_PATH = os.path.join(TRAINED_OUTPUT_FOLDER_NAME, CHUNKS_PATH)
TRAINED_QUESTION_EMBEDDINGS_PATH = os.path.join(TRAINED_OUTPUT_FOLDER_NAME, QUESTIONS_PATH)
TRAINED_TOP_K_ACCURACIES_OUTPUT = os.path.join(TRAINED_OUTPUT_FOLDER_NAME, TOP_K_ACCURACIES_PATH)

TRAINED_CHUNK_EMBEDDINGS_TSNE_PATH = os.path.join(TRAINED_OUTPUT_FOLDER_NAME, CHUNKS_TSNE_PATH)
TRAINED_QUESTION_EMBEDDINGS_TSNE_PATH = os.path.join(TRAINED_OUTPUT_FOLDER_NAME, QUESTIONS_TSNE_PATH)
TRAINED_COMBINE_EMBEDDINGS_TSNE_PATH = os.path.join(TRAINED_OUTPUT_FOLDER_NAME, COMBINE_TSNE_PATH)
