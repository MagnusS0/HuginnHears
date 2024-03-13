import sys
from pathlib import Path

# Add the project directory to sys.path
root_path = Path(__file__).parent.parent.parent
sys.path.append(str(root_path))

from src.models.nb_whisper_model.load_model import nb_whisper


def transcriber(audio_path):
    segments, _ = nb_whisper.transcribe(audio_path, 
                                        beam_size=5, 
                                        language='no', 
                                        chunk_length=28
                                    )

    transcribed_text = "\n".join([f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}" for segment in segments])

    return transcribed_text
    
if __name__ == "__main__":
    audio_path = 'test_files/king.mp3'
    print(transcriber(audio_path))
    