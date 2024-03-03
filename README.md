# Whisper-LLM Project

This project focuses on integrating a Whisper model fine-tuned on Norwegian for transcription and connecting it to a Large Language Model (LLM) with LangChain for summarization.

## Project Structure

```
whisper-llm-project
├── src
│   ├── whisper
│   │   └── norwegian_transcription.py
│   ├── llm
│   │   └── summarization.py
│   └── main.py
├── models
│   ├── whisper
│   │   └── norwegian_model
│   └── llm
│       └── langchain_model
├── tests
│   ├── test_transcription.py
│   └── test_summarization.py
├── requirements.txt
└── README.md
```

## Files

### `src/whisper/norwegian_transcription.py`

This file contains the code for integrating the Whisper model fine-tuned on Norwegian for transcription. It includes functions and classes for transcribing Norwegian audio.

### `src/llm/summarization.py`

This file contains the code for connecting to a Large Language Model (LLM) with LangChain for summarization. It includes functions and classes for generating summaries from text.

### `src/main.py`

This file is the entry point of the application. It uses the Whisper model for transcription and the LLM with LangChain for summarization to process audio and generate summaries.

### `models/whisper/norwegian_model`

This directory contains the Whisper model fine-tuned on Norwegian for transcription. It includes the necessary files and configurations for using the model.

### `models/llm/langchain_model`

This directory contains the Large Language Model (LLM) with LangChain for summarization. It includes the necessary files and configurations for using the model.

### `tests/test_transcription.py`

This file contains unit tests for the Whisper model's transcription functionality. It includes test cases for different scenarios and edge cases.

### `tests/test_summarization.py`

This file contains unit tests for the LLM with LangChain's summarization functionality. It includes test cases for different scenarios and edge cases.

### `requirements.txt`

This file lists the dependencies required for the project. It includes the necessary packages and versions for running the code.

## Usage

1. Install the required dependencies by running `pip install -r requirements.txt`.
2. Place the audio files you want to transcribe in a directory.
3. Update the necessary configurations in `src/whisper/norwegian_transcription.py` and `src/llm/summarization.py`.
4. Run `python src/main.py` to start the transcription and summarization process.
5. The transcriptions and summaries will be generated and saved in the specified output directory.

## Testing

To run the unit tests for the Whisper model's transcription functionality, execute `python -m unittest tests/test_transcription.py`.

To run the unit tests for the LLM with LangChain's summarization functionality, execute `python -m unittest tests/test_summarization.py`.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

This project utilizes the Whisper model for transcription and the Large Language Model (LLM) with LangChain for summarization. Special thanks to the developers and contributors of these models.

Please note that this project is for educational purposes only and should not be used for any commercial applications without proper authorization.