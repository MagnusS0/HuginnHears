import streamlit as st
import sys
import os
sys.path.append(os.getcwd() + '/')
import tempfile
import base64
from huginn_hears.main import WhisperTranscriber, MistralSummarizer, ExtractiveSummarizer
from prompt_templates import king_refine_template, king_prompt_template, en_prompt_template, en_refine_template


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
        f.write(audio_file.read()) # Write the file to the server

    return temp_dir, file_path


def main():
    st.title("Audio Transcription and Summarization")
    
    uploaded_file = st.file_uploader("Choose an audio file", type=['wav', 'mp3', 'm4a'])
    transcript = None  # Initialize transcript variable
    
    if uploaded_file is not None:
        # Save the uploaded audio file to the server
        _ , file_path = get_file_path(uploaded_file)
        st.success(f"File uploaded")
        
        # Initialize the transcriber and summarizer
        transcriber = WhisperTranscriber()
        extractive_summarizer = ExtractiveSummarizer()
        summarizer = MistralSummarizer(model_path='/home/magsam/llm_models/mistral-7b-instruct-v0.2.Q4_K_M.gguf', prompt_template=en_prompt_template, refine_template=en_refine_template)
        
        # Cache the transcribe and summarize functions to avoid re-running them
        @st.cache_data(persist=True, show_spinner=False)
        def cached_transcribe(audio_path):
            return transcriber.transcribe(audio_path)

        @st.cache_data(persist=True, show_spinner=False)
        def chached_extractive_summarize(transcript):
            return extractive_summarizer.extractive_summarize(transcript)

        @st.cache_data(persist=True, show_spinner=False)
        def cached_summarize(transcript):
            return summarizer.summarize(transcript)

        # Transcribe
        if st.button("Transcribe"):
            with st.spinner("Summarizing transcript..."):
                transcript = cached_transcribe(file_path)
            st.success("Transcription complete!")
            st.download_button(label="Download Transcript", data=transcript, file_name="transcript.txt", mime="text/plain", type='primary')
        
        # Summerize with or without extractive summarization
        if st.checkbox("Use Extractive Summarization"):
            if st.button("Summarize"):
                with st.spinner("Summarizing transcript, this may take a while..."):
                    # Ensure transcription is done before summarization
                    transcript = cached_transcribe(file_path)
                    summary_ex = chached_extractive_summarize(transcript)
                    summary = cached_summarize(summary_ex)
                # Display summary
                st.write("Summary:")
                st.write(summary)
                # Download summary 
                st.download_button(label="Download Summary", data=summary, file_name="summary.txt", mime="text/plain", type='primary')
        else:

            if st.button("Summarize"):
                with st.spinner("Summarizing transcript, this may take a while..."):
                    # Ensure transcription is done before summarization
                    transcript = cached_transcribe(file_path)
                    summary = cached_summarize(transcript)
                # Display summary
                st.write("Summary:")
                st.write(summary)
                # Download summary 
                st.download_button(label="Download Summary", data=summary, file_name="summary.txt", mime="text/plain", type='primary')


if __name__ == "__main__":
    main()
