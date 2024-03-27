from prompt_templates import (no_refine_template, no_prompt_template,
                              en_prompt_template, en_refine_template,
                              chatml_no_prompt_template, chatml_no_refine_template,
                              chatml_en_prompt_template, chatml_en_refine_template,
                              )
from huginn_hears.main import WhisperTranscriber, LLMSummarizer, ExtractiveSummarizer
import gc
import torch
import tempfile
import streamlit as st
import sys
import os
sys.path.append(os.getcwd() + '/')


def get_file_path(audio_file):
    """
    Save the audio file to the server and return the path.

    Args:
        audio_file (File): The uploaded audio file.

    Returns:
        str: The path to the saved audio file.
    """
    temp_dir = tempfile.mkdtemp()
    print(temp_dir)

    file_path = os.path.join(temp_dir, audio_file.name)

    with open(file_path, 'wb') as f:
        f.write(audio_file.read())  # Write the file to the server

    return temp_dir, file_path


# All prompts and refine templates
prompt_templates = {
    "Norwegian Prompt": no_prompt_template,
    "English Prompt": en_prompt_template,
    "Norwegian ChatML Prompt": chatml_no_prompt_template,
    "English ChatML Prompt": chatml_en_prompt_template,
}

refine_templates = {
    "Norwegian Refine": no_refine_template,
    "English Refine": en_refine_template,
    "Norwegian ChatML Refine": chatml_no_refine_template,
    "English ChatML Refine": chatml_en_refine_template,
}

# Initialize the transcript to None
if 'transcript' not in st.session_state:
    st.session_state.transcript = None



def main():
    st.title("Huginn Hears - Audio Summarization 🐦‍⬛")

    st.write("""Transcribe and summarize audio files using Huginn Hears on your local machine.
            Huginn Hears is a tool that uses NB-Whisper for transcription and any LLM from Huggingface you want to summerize the meeting. 
            It is optimized for Norwegian and summerizing meetings.""")
    
    # Notice about the initial download time for models
    st.info("Please note: The first run of Huginn Hears may be slower as it downloads all necessary models. This is a one-time process, and subsequent runs will be significantly faster.")
    # Get repo id and filename for the Hugging Face model
    mistral_model_path = st.text_input(
        "Enter the Hugging Face repo id", value="TheBloke/dolphin-2.6-mistral-7B-dpo-laser-GGUF")
    mistral_filename = st.text_input(
        "Enter the filename of the model", value="*Q4_K_M.gguf")
    # Check if the user has entered the model path and filename
    if not mistral_model_path or not mistral_filename:
        st.warning("Please enter the Hugging Face repo id and the filename of the model.")
        st.stop()

    else:
        col1, col2 = st.columns(2)
        # Select prompt and refine templates
        with col1:
            prompt_template = st.selectbox("Select prompt template", list(prompt_templates.keys()))
        with col2:   
            refine_template = st.selectbox("Select refine template", list(refine_templates.keys()))

        # Use the selected templates
        selected_prompt_template = prompt_templates[prompt_template]
        selected_refine_template = refine_templates[refine_template]

        # Advanced options
        with st.expander("**Advanced Options** 🛠️", expanded=False):
            # Adjust model parameters
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                n_ctx = st.slider("Context Size", min_value=0, max_value=8192, value=4096, step=256)
            with col2:
                max_tokens = st.slider("Max Tokens (Output)", min_value=0, max_value=1024, value=512, step=128)
            with col3:
                n_batch = st.slider("Batch Size", min_value=0, max_value=512, value=16, step=4)
            with col4:
                n_threads = st.slider("Threads", min_value=0, max_value=16, value=4, step=1)

            model_options = {
                "n_ctx": n_ctx,
                "max_tokens": max_tokens,
                "n_batch": n_batch,
                "n_threads": n_threads
            }

            # Custom prompt and refine templates
            st.info("You can write your own prompt and refine templates. NB: Make sure to include the placeholders `{text}` and `{existing_answer}` in the templates.")
            custom_prompt = st.text_area("Custom Prompt Template", placeholder="Summerize this text: {text}")
            custom_refine = st.text_area("Custom Refine Template", placeholder="Refine the summary: {existing_answer} with additional text: {text}")
            if custom_prompt:
                selected_prompt_template = custom_prompt
            if custom_refine:
                selected_refine_template = custom_refine


    uploaded_file = st.file_uploader(
        "Choose an audio file", type=['wav', 'mp3', 'm4a'])

    if uploaded_file is not None:
        # Save the uploaded audio file to the server
        _, file_path = get_file_path(uploaded_file)
        st.success(f"File uploaded")

        # Set language of transcript
        if st.toggle("Set language to English"):
            st.session_state.language = "en"
        else:
            st.session_state.language = "no"



        # Initialize the transcriber and summarizer
        transcriber = WhisperTranscriber(language=st.session_state.language)
        extractive_summarizer = ExtractiveSummarizer()
        summarizer = LLMSummarizer(repo_id=mistral_model_path, 
                                       filename=mistral_filename,
                                       prompt_template=selected_prompt_template, 
                                       refine_template=selected_refine_template, model_options=model_options)

        # Cache the transcribe and summarize functions to avoid re-running them
        @st.cache_data(persist=True, show_spinner="Transcribing audio...")
        def cached_transcribe(audio_path):
            return transcriber.transcribe(audio_path)

        @st.cache_data(persist=True, show_spinner="Extractive summarization...")
        def cached_extractive_summarize(transcript):
            return extractive_summarizer.extractive_summarize(transcript)

        @st.cache_data(persist=True, show_spinner="Summarizing transcript, this may take a while...")
        def cached_summarize(transcript):
            return summarizer.summarize(transcript)

        # Transcribe
        if st.button("Transcribe"):
            st.session_state.transcript = cached_transcribe(file_path)
            st.success("Transcription complete!")
        if st.session_state.transcript:
            st.download_button(label="Download Transcript", data=st.session_state.transcript,
                            file_name="transcript.txt", mime="text/plain", type='primary')

        # Summerize with or without extractive summarization
        if st.toggle("Use Extractive Summarization"):
            if st.button("Summarize"):
                # Ensure transcription is done before summarization
                if st.session_state.transcript is None:
                    st.session_state.transcript = cached_transcribe(file_path)
                # Else use the cached transcript
                transcript = st.session_state.transcript
                summary_ex = cached_extractive_summarize(transcript)
                # Clear GPU memory becasue context manager does not work correctly
                torch.cuda.empty_cache()
                gc.collect()
                summary = cached_summarize(summary_ex)
                # Display summary
                st.write("**Summary**:")
                st.write(summary)
                # Download summary
                st.download_button(label="Download Summary", data=summary,
                                    file_name="summary.txt", mime="text/plain", type='primary')
        else:

            if st.button("Summarize"):
                # Ensure transcription is done before summarization
                if st.session_state.transcript is None:
                    st.session_state.transcript = cached_transcribe(file_path)
                # Else use the cached transcript
                transcript = st.session_state.transcript
                summary = cached_summarize(transcript)
                # Display summary
                st.write("**Summary**:")
                st.write(summary)
                # Download summary
                st.download_button(label="Download Summary", data=summary,
                                    file_name="summary.txt", mime="text/plain", type='primary')


if __name__ == "__main__":
    main()
