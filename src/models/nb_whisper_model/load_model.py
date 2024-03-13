# Load the model from the HuggingFace model hub 
import torch
from faster_whisper import WhisperModel

def load_model(model_size='small', compute_type = "float16", device = 'cuda' if torch.cuda.is_available() else 'cpu' ):
    model = WhisperModel(f"NbAiLab/nb-whisper-{model_size}", 
                         device=device, 
                         compute_type=compute_type, 
                         cpu_threads=4, 
                         num_workers=6
                    )
    return model


nb_whisper = load_model()