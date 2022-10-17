import logging
from pathlib import Path

import whisper
from tqdm import tqdm

WORKDIR = Path(__file__).parent
DATADIR = WORKDIR / 'data'
TRANSCRIPTIONDIR = WORKDIR / 'transcription'

logger = logging.getLogger(__name__)

class Whisper:
    """
    If ffmpeg gives error
    pip uninstall ffmpeg
    pip uninstall ffmpeg-python

    and install ffmpeg-python again with :
    pip install ffmpeg-python
    """
    def __init__(self, model="large"):
        self.model = whisper.load_model(model)
        logger.info("Whisper model loaded")

    def transcribe(self, audio_path):
        result = self.model.transcribe(audio_path)
        return result["text"]
    
    def transcribe_folder(self, folder_path):
        folder_path = Path(folder_path)
        filelist = list(folder_path.rglob('*.*'))
        for filepath in tqdm(filelist, leave=False):
            text = self.transcribe(str(filepath))
            transcription_path = Path(str(filepath).replace(str(DATADIR), str(TRANSCRIPTIONDIR)).replace(".wav", ".txt"))
            transcription_path.parent.mkdir(parents=True, exist_ok=True)
            with open(transcription_path, 'w') as f:
                f.write(text)
        logger.info(f"Transcriptions saved to {TRANSCRIPTIONDIR}")



if __name__ == '__main__':
    whisper = Whisper()
    whisper.transcribe_folder(DATADIR / 'NH2')
