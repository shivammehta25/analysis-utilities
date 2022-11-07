#!/usr/bin/env python
import argparse
from pathlib import Path

import whisper
from tqdm import tqdm

WORKDIR = Path(__file__).parent
DATADIR = WORKDIR / 'LJ_Valid_data'
TRANSCRIPTIONDIR = WORKDIR / 'LJ_Valid_transcription'

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
        print("[+] Whisper model loaded")

    def transcribe(self, audio_path, lang="en"):
        result = self.model.transcribe(audio_path, language=lang)
        return result["text"]
    
    def transcribe_folder(self, input_dir, output_dir, exception=None):
        if exception:
            if isinstance(exception, list):
                exception = set(exception)
            if isinstance(exception, str):
                exception = set([exception])

        print(f"Running whisper over: {input_dir}")
        input_dir = Path(input_dir)
        filelist = list(input_dir.rglob('*.*'))
        for filepath in tqdm(filelist, leave=False):
            if exception and filepath.parent.parent.name in exception:    # change here the location to exclude
                print(f"\r[!] Skipping: {exception}")
                continue
            transcription_path = Path(str(filepath).replace(str(DATADIR), str(output_dir)).replace(".wav", ".txt"))
            transcription_path.parent.mkdir(parents=True, exist_ok=True)
            
            if transcription_path.exists():
                print(f"\r[!] Skipping: {transcription_path} it already exists!")
                continue
            
            text = self.transcribe(str(filepath))

            with open(transcription_path, 'w') as f:
                f.write(text)
        print(f"[+] Transcriptions saved to {output_dir}")
        
        
def main():
    model_choices = ["large", "medium", "small"]
    parser = argparse.ArgumentParser(description='Run whisper on a folder')
    parser.add_argument("-m", "--model", default=model_choices[1], help="Model to use", choices=model_choices)
    parser.add_argument("-i", "--input", help="Input folder", required=True)
    parser.add_argument("-o", "--output", help="Output folder", required=True)
    parser.add_argument('-e','--exceptions', nargs='+', help='Subfolders to exclude', default=None)
    args = parser.parse_args()
    print(args)
    whisper = Whisper(model=args.model)
    whisper.transcribe_folder(args.input, args.output, exception=args.exceptions)

if __name__ == '__main__':
    main()