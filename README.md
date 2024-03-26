# Huginn Hears
Huginn Hears is a Python application designed to transcribe speech and summarize it in Norwegian. It is meant to be used locally and ran on a single machine. The application is built using the [Streamlit](https://streamlit.io/) framework for the user interface, [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper) for speech-to-text transcription, [llmlingua-2](https://github.com/microsoft/LLMLingua) for compressing the transcribed text and [llama-ccp-python](https://github.com/abetlen/llama-cpp-python) for summarization. You are free to 

## Features
- Transcribes speech into text.
- Summarizes the transcribed text.
- Supports both English and Norwegian languages.

## Installation
This project uses [Poetry](https://python-poetry.org/) for dependency management. Before installing the project dependencies, it's essential to set up certain environment variables required by [llama-ccp-python](https://github.com/abetlen/llama-cpp-python).

### Setting Environment Variables for llama-cpp-python
`llama-cpp-python` requires specific environment variables to be set up in your system to function correctly. Follow the instructions in their repo to get the correct variables for your system. https://github.com/abetlen/llama-cpp-python 
#### Commen errors 🤯
<details>
<summary> Soulutions from llama-cpp-python </summary>
    
### Windows Notes

<details>
<summary>Error: Can't find 'nmake' or 'CMAKE_C_COMPILER'</summary>

If you run into issues where it complains it can't find `'nmake'` `'?'` or CMAKE_C_COMPILER, you can extract w64devkit as [mentioned in llama.cpp repo](https://github.com/ggerganov/llama.cpp#openblas) and add those manually to CMAKE_ARGS before running `pip` install:

```ps
$env:CMAKE_GENERATOR = "MinGW Makefiles"
$env:CMAKE_ARGS = "-DLLAMA_OPENBLAS=on -DCMAKE_C_COMPILER=C:/w64devkit/bin/gcc.exe -DCMAKE_CXX_COMPILER=C:/w64devkit/bin/g++.exe"
```

See the above instructions and set `CMAKE_ARGS` to the BLAS backend you want to use.
</details>

### MacOS Notes

Detailed MacOS Metal GPU install documentation is available at [docs/install/macos.md](https://llama-cpp-python.readthedocs.io/en/latest/install/macos/)

<details>
<summary>M1 Mac Performance Issue</summary>

Note: If you are using Apple Silicon (M1) Mac, make sure you have installed a version of Python that supports arm64 architecture. For example:

```bash
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh
bash Miniforge3-MacOSX-arm64.sh
```

Otherwise, while installing it will build the llama.cpp x86 version which will be 10x slower on Apple Silicon (M1) Mac.
</details>

<details>
<summary>M Series Mac Error: `(mach-o file, but is an incompatible architecture (have 'x86_64', need 'arm64'))`</summary>

Try installing with

```bash
CMAKE_ARGS="-DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_APPLE_SILICON_PROCESSOR=arm64 -DLLAMA_METAL=on" pip install --upgrade --verbose --force-reinstall --no-cache-dir llama-cpp-python
```
</details>
</details>

### Installing Project Dependencies
After setting up the required environment variables, you can proceed to install the project.
Ensure you have Poetry installed on your system. Then, run the following command in the project root directory:
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
- [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper) for speech-to-text transcription.
- [llmlingua-2](https://github.com/microsoft/LLMLingua) for prompt compression.
- [llama-ccp-python](https://github.com/abetlen/llama-cpp-python) to run LLMs locally and on CPUs.
    - [llama-ccp](https://github.com/ggerganov/llama.cpp)
- [cx_Freeze](https://cx-freeze.readthedocs.io/) for building executables.
- [Langchain](https://www.langchain.com/) for controlling the prompt-response flows.
- [Streamlit](https://streamlit.io/) for building the UI.

Big thanks to all the contributors to these open-source projects!

In addition, the following models were used:
- [Nasjonalbiblioteket AI Lab](https://huggingface.co/NbAiLab/nb-whisper-small) NB-Whisper.
- [Microsoft](https://huggingface.co/microsoft/llmlingua-2-xlm-roberta-large-meetingbank) LLMLingua-2.
- [TheBloke](https://huggingface.co/TheBloke) for all sorts quantisized models.

### Papers
You can read more about these models in these papers:

- [LLMLingua-2: Data Distillation for Efficient and Faithful Task-Agnostic Prompt Compression](https://arxiv.org/abs/2403.12968)

- [Whispering in Norwegian: Navigating Orthographic and Dialectic Challenges](https://arxiv.org/abs/2402.01917)


## License
This project is licensed under the Appache 2.0 License. See the `LICENSE` file for more information.

