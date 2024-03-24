# Huginn Hears
Huginn Hears is a Python application designed to transcribe speech and summarize it in Norwegian. It is meant to be used locally and ran on a single machine. The application is built using the [Streamlit](https://streamlit.io/) framework for the user interface, [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper) for speech-to-text transcription, [llmlingua-2](https://github.com/microsoft/LLMLingua) for compressing the transcribed text and [llama-ccp-python](https://github.com/abetlen/llama-cpp-python) for summarization. You are free to 

## Features
- Transcribes speech into text.
- Summarizes the transcribed text.
- Supports both English and Norwegian languages.

## Installation
This project uses [Poetry](https://python-poetry.org/) for dependency management. To install the project, first install Poetry, then run the following command in the project root:
```bash
poetry install
```
This will install all the necessary dependencies as defined in the `pyproject.toml` file.

## Usage
To run the application, use the following command:
```bash
python streamlit_app/run.py
```
This will start the Streamlit server and the application will be accessible at `localhost:8501`.

## Building
To build the project into an executable, use the `setup.py` script with [cx_Freeze](https://cx-freeze.readthedocs.io/):
```bash
python setup.py build
```
This will create an executable in the `build` directory.

## Acknowledgements
This project build on a lot of great work done by others. The following projects were used:
- [Faster-Whisper]((https://github.com/SYSTRAN/faster-whisper))
- [llmlingua-2](https://github.com/microsoft/LLMLingua)
- [llama-ccp-python](https://github.com/abetlen/llama-cpp-python)
    - [llama-ccp](https://github.com/ggerganov/llama.cpp)
- [Poetry](https://python-poetry.org/)
- [cx_Freeze](https://cx-freeze.readthedocs.io/)
- [Langchain](https://www.langchain.com/)
- [Streamlit](https://streamlit.io/)

Big thanks to all the contributors to these open-source projects!

In addition, the following models were used:
- [Nasjonalbiblioteket AI Lab](https://huggingface.co/NbAiLab/nb-whisper-small)
- [Microsoft](https://huggingface.co/microsoft/llmlingua-2-xlm-roberta-large-meetingbank)
- [TheBloke](https://huggingface.co/TheBloke) for all sorts quantisized models.

## License
This project is licensed under the Appache 2.0 License. See the `LICENSE` file for more information.

