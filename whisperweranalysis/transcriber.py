import logging
from pathlib import Path

import whisper
from tqdm import tqdm

WORKDIR = Path(__file__).parent
DATADIR = WORKDIR / 'LJ_Valid_data'
TRANSCRIPTIONDIR = WORKDIR / 'LJ_Valid_transcription'

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

    def transcribe(self, audio_path, lang="en"):
        result = self.model.transcribe(audio_path, language=lang)
        return result["text"]
    
    def transcribe_folder(self, folder_path, transcription_datadir=TRANSCRIPTIONDIR, exception=None):
        if exception:
            if isinstance(exception, list):
                exception = set(exception)
            if isinstance(exception, str):
                exception = set([exception])

        logger.info(f"Running whisper over: {folder_path}")
        folder_path = Path(folder_path)
        filelist = list(folder_path.rglob('*.*'))
        for filepath in tqdm(filelist, leave=False):
            if exception and filepath.parent.parent.name in exception:    # change here the location to exclude
                print(f"\r Skipping: {exception}")
                continue
            transcription_path = Path(str(filepath).replace(str(DATADIR), str(transcription_datadir)).replace(".wav", ".txt"))
            transcription_path.parent.mkdir(parents=True, exist_ok=True)
            
            if transcription_path.exists():
                print(f"\r Skipping: {transcription_path}")
                continue
            
            text = self.transcribe(str(filepath))

            with open(transcription_path, 'w') as f:
                f.write(text)
        logger.info(f"Transcriptions saved to {transcription_datadir}")



if __name__ == '__main__':
    whisper = Whisper("medium")
    whisper.transcribe_folder(DATADIR, exception=["NS", "OverFlow"])
