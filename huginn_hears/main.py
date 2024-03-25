import gc
import torch
from faster_whisper import WhisperModel
from contextlib import contextmanager

from llmlingua import PromptCompressor
from transformers import AutoTokenizer

from huginn_hears.utils import refine_summary
from huginn_hears.utils import CustomLlamaCpp

from langchain.schema.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


class WhisperTranscriber:
    """
    A class for transcribing audio using the Faster-Whisper ASR model.

    Args:
        model_size (str): The size of the Whisper model. Defaults to 'small'.
        compute_type (str): The compute type for the model. Defaults to 'float16'.

    Methods:
        load_model: A context manager for loading the Whisper model.
        transcribe: Transcribes the audio using the loaded model.

    """

    def __init__(self, model_size='small', compute_type="float16"):
        self.model_size = model_size
        self.compute_type = compute_type
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None

    @contextmanager
    def load_model(self, cpu_threads=4, num_workers=6):
        """
        A context manager for loading and unloading the Whisper model.

        Args:
            cpu_threads (int): The number of CPU threads to use. Defaults to 4.
            num_workers (int): The number of workers for data loading. Defaults to 6.

        Returns:
            WhisperModel: The loaded Whisper ASR model.

        """
        self.model = WhisperModel(f"NbAiLab/nb-whisper-{self.model_size}",  # Model needs to be compatible with faster-whisper's Ctranselate2 engine
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
        """
        Transcribes the audio using the loaded Whisper model.

        Args:
            audio_path (str): The path to the audio file.
            timestamp (bool): Whether to include timestamps in the transcribed text. Defaults to False.

        Returns:
            str: The transcribed text.

        """
        with self.load_model() as model:
            segments, _ = model.transcribe(audio_path,
                                           beam_size=5,
                                           task="transcribe",
                                           language='no',  # Declearing language is faster but optional
                                           chunk_length=28)  # 28 provides better results
            if timestamp:
                transcribed_text = "\n".join(
                    [f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}" for segment in segments])
                return transcribed_text

            transcribed_text = "\n".join(
                [segment.text for segment in segments])
            return transcribed_text


class ExtractiveSummarizer:
    """
    A class for summarizing documents using LLMLingua-2 compression.

    Args:
        model_name (str): The name of the extractive summarization model from HF Hub.
        rate (float): The forced compression rate of the document. Defaults to 0.5.

    """

    def __init__(self, model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank", rate=0.6):
        self.model_name = model_name
        self.rate = rate
        self.model = None

    @contextmanager
    def load_extractive_summarizer(self):
        """
        Context manager for loading and unloading the extractive summarizer model.

        Yields:
            Summarizer: The loaded extractive summarizer model.
        """
        model = PromptCompressor(
            model_name=self.model_name,
            use_llmlingua2=True
        )
        try:
            yield model
        finally:
            del model
            torch.cuda.empty_cache()
            gc.collect()

    def _compressed_summary(self, text, model):
        summary = model.compress_prompt_llmlingua2(
            text,
            rate=self.rate,
            force_tokens=['\n', '.', '!', '?'],
            chunk_end_tokens=['.', '\n'],
            return_word_label=False,
            drop_consecutive=True
        )
        return summary['compressed_prompt']

    def extractive_summarize(self, document: str) -> str:
        """
        Summarizes a document using an extractive summarizer model.

        Args:
            document (str): The document to summarize.

        Returns:
            str: The summarized document.
        """

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer,
            chunk_size=512)

        documents = splitter.split_text(document)

        with self.load_extractive_summarizer() as model:
            summaries = [self._compressed_summary(
                chunk, model) for chunk in documents]
            final_summary = ' '.join(summaries)
            return final_summary


class MistralSummarizer:
    """
    A class for summarizing documents using the Mistral model.

    Args:
        model_path (str): The path to the Mistral model.
        text_splitter (TextSplitter, optional): The text splitter to use for splitting documents into chunks. Defaults to RecursiveCharacterTextSplitter.
        prompt_template (str, optional): The prompt template to use for generating prompts. Defaults to None.
        refine_template (str, optional): The refine template to use for refining summaries. Defaults to None.
    """

    def __init__(self, repo_id: str, filename: str, text_splitter=RecursiveCharacterTextSplitter, prompt_template: str = None, refine_template: str = None):
        self.repo_id = repo_id
        self.filename = filename
        self.layers = -1 if torch.cuda.is_available() else None
        self.model = None
        self.text_splitter = text_splitter(chunk_size=2048)
        self.prompt_template = PromptTemplate.from_template(prompt_template)
        self.refine_template = PromptTemplate.from_template(refine_template)

    @contextmanager
    def load_model(self, n_ctx=4096, max_tokens=512, n_batch=512, n_threads=6, temperature=0.2):
        """
        Context manager for loading and unloading the Mistral model.

        Args:
            n_ctx (int, optional): The context size for the model. Defaults to 4096.
            max_tokens (int, optional): The maximum number of tokens to generate. Defaults to 1024.
            n_batch (int, optional): The batch size for model inference. Defaults to 512.
            n_threads (int, optional): The number of threads to use for model inference. Defaults to 4.
            temperature (float, optional): The temperature for sampling from the model. Defaults to 0.2.

        Yields:
            LlamaCpp: The loaded Mistral model.
        """
        self.model = CustomLlamaCpp(
            repo_id=self.repo_id,
            filename=self.filename,
            n_gpu_layers=self.layers,
            n_ctx=n_ctx,
            max_tokens=max_tokens,
            n_batch=n_batch,
            n_threads=n_threads,
            temperature=temperature,
            top_p=0.9,
            repeat_penalty=1.18,  # Trying to avoid repeating the same words
            verbose=True,
            chat_format="chatml",
        )
        try:
            yield self.model
        finally:
            del self.model
            torch.cuda.empty_cache()
            gc.collect()

    def summarize(self, document) -> str:
        """
        Summarizes a document using either the full context or a refine chain.

        Args:
            document (str): The document to summarize.

        Returns:
            str: The summarized document.
        """
        docs = [Document(page_content=document)]
        output_parser = StrOutputParser()

        with self.load_model() as model:

            try:  # Trying to use the full context
                chain = (
                    self.prompt_template
                    | model
                    | output_parser
                    )
                result = chain.invoke({"text": docs})
                return result
            except: # If the full context fails, use the refine chain
                split_docs = self.text_splitter.split_documents(docs)
                refine_chain = (
                    self.refine_template
                    | model
                    | output_parser
                )
                # Combine the chains
                summary = refine_summary(chain=chain, refine_chain=refine_chain, split_docs=split_docs)
            
                return summary


class AudioSummarizationPipeline:
    """
    A class representing an audio summarization pipeline.

    Args:
        audio_path (str): The path to the audio file.
        transcriber (WhisperTranscriber): An instance of the WhisperTranscriber class for transcribing the audio.
        summarizer (MistralSummarizer): An instance of the MistralSummarizer class for summarizing the transcript.

    Raises:
        TypeError: If transcriber is not an instance of WhisperTranscriber.
        TypeError: If summarizer is not an instance of MistralSummarizer.
    """

    def __init__(self, audio_path, transcriber: WhisperTranscriber, summarizer: MistralSummarizer, extractor: ExtractiveSummarizer):
        if not isinstance(transcriber, WhisperTranscriber):
            raise TypeError(
                f'transcriber must be an instance of WhisperTranscriber, got {type(transcriber)} instead')
        if not isinstance(summarizer, MistralSummarizer):
            raise TypeError(
                f'summarizer must be an instance of MistralSummarizer, got {type(summarizer)} instead')
        if not isinstance(extractor, ExtractiveSummarizer):
            raise TypeError(
                f'extractor must be an instance of ExtractiveSummarizer, got {type(extractor)} instead')

        self.audio_path = audio_path
        self.transcriber = transcriber
        self.summarizer = summarizer
        self.extractor = extractor

    def run(self, extractive_summary=False):
        """
        Runs the audio summarization pipeline.

        Transcribes the audio using the transcriber, summarizes the transcript using the summarizer,
        and prints the summary.

        Returns:
            str: The summarized document.

        Raises:
            Exception: If an error occurs during the transcription or summarization process.
        """
        try:
            transcript = self.transcriber.transcribe(self.audio_path)
            if extractive_summary:
                transcript = self.extractor.extractive_summarize(transcript)
                torch.cuda.empty_cache() # Because momory release is not working in the context manager
                gc.collect()
            summary = self.summarizer.summarize(transcript)
            print(summary)
        except Exception as e:
            print(f"An error occurred: {e}")


if __name__ == "__main__":
    # Define the prompt and refine templates
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
    summarizer = MistralSummarizer(repo_id="TheBloke/dolphin-2.6-mistral-7B-dpo-laser-GGUF", filename='*Q4_K_M.gguf',
                                   prompt_template=prompt_template, refine_template=refine_template)
    extractor = ExtractiveSummarizer()
    audio_path = '/home/magsam/workspace/huginn-hears/test_files/king.mp3'
    pipeline = AudioSummarizationPipeline(
        audio_path, transcriber, summarizer, extractor)
    pipeline.run(extractive_summary=False)
