import sys
from pathlib import Path
import asyncio

# Add the project directory to sys.path
root_path = Path(__file__).parent.parent.parent
sys.path.append(str(root_path))

from src.models.mistral_7b_instruct.load_model import mistral_instruct

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import LLMChain

class DocumentSummarizer:
    """
    A class to summarize documents using language model chains.

    Attributes:
        llm (mistral_instruct): The language model to be used.
        text_splitter (RecursiveCharacterTextSplitter): The splitter to chunk large texts.
        prompt_template (str): Template for initial summary generation.
        refine_template (str): Template for refining summaries.
    """

    def __init__ (self, llm=mistral_instruct, text_splitter=RecursiveCharacterTextSplitter, prompt_template:str=None, refine_template:str=None):
        self.llm = llm
        self.text_splitter = text_splitter(chunk_size=1024)
        self.prompt_template = PromptTemplate.from_template(prompt_template)
        self.refine_template = PromptTemplate.from_template(refine_template)

    
    async def summarize(self, document: str) -> str:
        """
        Summarizes a given document using a summarization chain.

        Args:
            document (str): The path to the document to be summarized.

        Returns:
            str: The summarized text.
        """
        loader = TextLoader(document)
        docs = loader.load()

        # If the document is less than 3000 tokens, use a simple chain
        num_tokens = len(docs[0].page_content.split())

        # Chain will take the promt and entire document as input
        if num_tokens < 3000:
            llm_chain = LLMChain(llm=self.llm, prompt=self.prompt_template)
            result = llm_chain.invoke({"text": docs})
            return result

        # If the document is more than 3000 tokens it will be processed as chunks
        split_docs = self.text_splitter.split_documents(docs)
        chain = load_summarize_chain(
            llm=mistral_instruct,
            chain_type="refine",
            question_prompt=self.prompt_template,
            refine_prompt=self.refine_template,
            return_intermediate_steps=False,
            input_key="input_documents",
            output_key="output_text",
        )

        result = await chain.ainvoke({"input_documents": split_docs})
        return result


if __name__ == "__main__":
    prompt_template =""" 
    You are an expert agent in information extraction and summarization.
    Given a meeting transcription generate a summary for record-keeping. 

    FORMAT:
    Title: 
    Key points:  
    Key Decisions:  
    Action Items: 

    The meeting transcript is as follows:
    ----------------
    {text}
    ----------------
    Focus solely on the information presented in the transcript. 
    Aim for clarity and brevity to ensure the summary is useful for attendees and those unable to attend.
    VERY IMPORTANT : Do not mention yourself, only output the meeting summary. No intro, no other text
    """
    refine_template = """
    Your task is to refine the initial summary based on additional context.
    Below is the existing summary:
    {existing_answer}
    Now, we'll provide more context from the meeting transcript:
    ----------------
    {text}
    ----------------
    Refine the summary to incorporate the new context, or maintain the original if the context does not add value. 
    FORMAT:
    Title: 
    Key points:
    Key Decisions: 
    Action Items: 

    Aim for clarity and brevity.
    VERY IMPORTANT : Do not mention yourself, only output the meeting summary. No intro, no other text
    """
    summarizer = DocumentSummarizer(llm=mistral_instruct, prompt_template=prompt_template, refine_template=refine_template)
    document = "/home/magsam/workspace/huginn-hears/test_files/gpt_transcript.txt"
    summary = asyncio.run(summarizer.summarize(document))
    print(summary)
