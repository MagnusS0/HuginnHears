from langchain_community.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler
import torch

callback_manager = CallbackManager([AsyncIteratorCallbackHandler()])

def load_model():
    layers = -1 if torch.cuda.is_available() else None
    model = LlamaCpp(
        model_path = '/home/magsam/llm_models/mistral-7b-instruct-v0.2.Q4_K_M.gguf',
        n_gpu_layers=layers,
        n_ctx=4096,
        max_tokens=1024,
        n_batch=16,
        n_threads=6,
        temperature=0,
        callback_manager=callback_manager,
        verbose=True # Verbose is required to pass to the callback manager
    )

    return model

mistral_instruct = load_model()

