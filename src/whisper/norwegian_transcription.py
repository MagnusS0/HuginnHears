import sys
from pathlib import Path

# Add the project directory to sys.path
root_path = Path(__file__).parent.parent.parent
sys.path.append(str(root_path))

from src.models.nb_whisper_model.load_model import nb_whisper


class WhisperTranscriber:
    """
    A class that represents a transcriber for Whisper audio models.

    Args:
        model (WhisperModel): The Whisper audio model to use for transcription. 
        language (str): The language code for the transcription. Defaults to 'no' (Norwegian).
    """

    def __init__(self, model=nb_whisper, language='no'):
        self.model = model
        self.language = language

    def transcribe(self, audio_path, chunk_length_s=28, return_timestamps=True, num_beams=5):
        """
        Transcribes the audio file at the given path.

        Args:
            audio_path (str): The path to the audio file to transcribe.
            chunk_length_s (int): The length of each audio chunk in seconds.
            return_timestamps (bool): Whether to return timestamps along with the transcription.
            num_beams (int): Increases the accuracy, but also the computation time and memory usage. Defaults to 5.

        Returns:
            transcription (dict): A dictionary containing the transcription results. The dictionary includes the following keys:
                    - 'text' (str): A string containing the complete transcription of the audio file.
                    - 'chunks' (list): A list of dictionaries, each representing a transcribed segment of the audio. Each dictionary in the list includes:
                        - 'timestamp' (tuple): A tuple representing the start and end times (in seconds) of the audio segment.
                        - 'text' (str): The transcribed text of the audio segment.
        """
        transcription = self.model(audio_path,
                                   chunk_length_s=chunk_length_s,
                                   return_timestamps=return_timestamps,
                                   generate_kwargs={'num_beams': num_beams,
                                                    'task': 'transcribe',
                                                    'language': self.language})
        return transcription['text']
    
if __name__ == "__main__":
    transcriber = WhisperTranscriber()
    audio_path = 'test_files/king.mp3'
    print(transcriber.transcribe(audio_path))
    