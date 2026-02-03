import torch

CORPUS_PATH = r"C:\Users\Jernej\Documents\karpathy_course\spellchecker\github-typo-corpus.v1.0.0.jsonl"
USE_DATASET = "brown"
MAX_SAMPLES = 50000
MAX_LEN = 20
BATCH_SIZE = 128
MAX_STEPS = 20000
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
