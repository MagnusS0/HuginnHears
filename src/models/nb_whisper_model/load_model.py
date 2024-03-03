# Load the model that can be imported in other scripts
from transformers import pipeline
import torch

def load_model(model_size='small'):
    device = 0 if torch.cuda.is_available() else -1
    model = pipeline("automatic-speech-recognition", f"NbAiLabBeta/nb-whisper-{model_size}", device=device)
    return model

nb_whisper = load_model()