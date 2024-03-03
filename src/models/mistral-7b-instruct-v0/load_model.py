from langchain_community.llms import LlamaCpp
import torch

def load_model(streaming=False):
    layers = -1 if torch.cuda.is_available() else None
    model = LlamaCpp(
        model_path = '/home/magsam/llm_models/mistral-7b-instruct-v0.1.Q4_0.gguf',
        n_gpu_layers=layers,
        n_batch=512,
        streaming=streaming
    )

    return model

mistral_instruct = load_model()

