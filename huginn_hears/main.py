import gc
import torch
from faster_whisper import WhisperModel
from contextlib import contextmanager

from langchain_community.llms import LlamaCpp
from langchain.schema.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import LLMChain


class WhisperTranscriber:
    def __init__(self, model_size='small', compute_type="float16"):
        self.model_size = model_size
        self.compute_type = compute_type
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None

    @contextmanager
    def load_model(self, cpu_threads=4, num_workers=6):
        self.model = WhisperModel(f"NbAiLab/nb-whisper-{self.model_size}",
                                  device=self.device,
                                  compute_type=self.compute_type,
                                  cpu_threads=cpu_threads,
                                  num_workers=num_workers)
        try:
            yield self.model
        finally:
            del self.model
            torch.cuda.empty_cache()
            gc.collect()

    def transcribe(self, audio_path, timestamp=False):
        with self.load_model() as model:
            segments, _ = model.transcribe(audio_path,
                                           beam_size=5,
                                           language='no',
                                           chunk_length=28)
            if timestamp:
                transcribed_text = "\n".join(
                    [f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}" for segment in segments])
                return transcribed_text

            transcribed_text = "\n".join([segment.text for segment in segments])
            return transcribed_text


class MistralSummarizer:
    def __init__(self, model_path, text_splitter=RecursiveCharacterTextSplitter, prompt_template: str = None, refine_template: str = None):
        self.model_path = model_path
        self.layers = -1 if torch.cuda.is_available() else None
        self.model = None
        self.text_splitter = text_splitter(chunk_size=1024)
        self.prompt_template = PromptTemplate.from_template(prompt_template)
        self.refine_template = PromptTemplate.from_template(refine_template)

    @contextmanager
    def load_model(self, n_ctx=4096, max_tokens=1024, n_batch=512, n_threads=4, temperature=0.2):
        self.model = LlamaCpp(
            model_path=self.model_path,
            n_gpu_layers=self.layers,
            n_ctx=n_ctx,
            max_tokens=max_tokens,
            n_batch=n_batch,
            n_threads=n_threads,
            temperature=temperature,
            top_p = 0.9,
            repeat_penalty=1.18,
            verbose=True
        )
        try:
            yield self.model
        finally:
            del self.model
            torch.cuda.empty_cache()
            gc.collect()

    def summarize(self, document: str) -> str:
        
        docs = [Document(page_content=document)]

        num_tokens = len(docs[0].page_content.split())

        with self.load_model() as model:

            if num_tokens < 2048:
                llm_chain = LLMChain(llm=model, prompt=self.prompt_template)
                result = llm_chain.invoke({"text": docs})
                return result

            split_docs = self.text_splitter.split_documents(docs)
            chain = load_summarize_chain(
                llm=model,
                chain_type="refine",
                question_prompt=self.prompt_template,
                refine_prompt=self.refine_template,
                return_intermediate_steps=False,
                input_key="input_documents",
                output_key="output_text",
            )

            result = chain.invoke({"input_documents": split_docs})
            return result


class AudioSummarizationPipeline:
    def __init__(self, audio_path, transcriber: WhisperTranscriber, summarizer: MistralSummarizer):
        if not isinstance(transcriber, WhisperTranscriber):
            raise TypeError(f'transcriber must be an instance of WhisperTranscriber, got {type(transcriber)} instead')
        if not isinstance(summarizer, MistralSummarizer):
            raise TypeError(f'summarizer must be an instance of MistralSummarizer, got {type(summarizer)} instead')
        
        self.audio_path = audio_path
        self.transcriber = transcriber
        self.summarizer = summarizer

    def run(self):
        try:
            transcript = self.transcriber.transcribe(self.audio_path)
            summary = self.summarizer.summarize(transcript)
            print(summary)
        except Exception as e:
            # Log the error, or print a user-friendly message
            print(f"An error occurred: {e}")




if __name__ == "__main__":
    # Define your prompt and refine templates
    prompt_template = """
    <s>[INST] Du er en ekspertagent i informasjonsutvinning og sammendrag.
    Gitt en tale transkripsjon, generer et sammendrag for arkivering.

    Transkripsjonen er som følger:
    ----------------
    {text}
    ----------------
    Fokuser kun på informasjonen presentert i transkripsjonen.
    Sikte på klarhet og korthet for å sikre at sammendraget er nyttig. Skriv på norsk.
    SVÆRT VIKTIG: Ikke nevn deg selv, kun skriv sammendraget. Ingen intro, ingen annen tekst. [/INST]
    """
    refine_template = """
    <s>[INST] Din oppgave er å forbedre dette sammendraget basert på ytterligere kontekst.
    Nedenfor er det eksisterende sammendraget:
    {existing_answer}
    Her er mer informasjon som kan hjelpe deg med å forbedre sammendraget:
    ----------------
    {text}
    ----------------
    Opdater sammendraget slik at det bruker den nye informasjonen, eller behold det originale hvis den nye informasjonen ikke tilfører verdi.
    Sikte på klarhet og korthet. Skriv på norsk.
    SVÆRT VIKTIG: Ikke nevn deg selv, kun skriv sammendraget. Ingen intro, ingen annen tekst [/INST]
    """
    transcriber = WhisperTranscriber()
    summarizer = MistralSummarizer(model_path='/home/magsam/llm_models/mistral-7b-instruct-v0.2.Q4_K_M.gguf', prompt_template=prompt_template, refine_template=refine_template)
    audio_path = '/home/magsam/workspace/huginn-hears/test_files/Kongens nyttårstale 2023.m4a'
    pipeline = AudioSummarizationPipeline(audio_path, transcriber, summarizer)
    pipeline.run()