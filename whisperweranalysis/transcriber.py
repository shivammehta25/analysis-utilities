#!/usr/bin/env python
import argparse
from pathlib import Path

import whisper
from tqdm import tqdm

WORKDIR = Path(__file__).parent
DATADIR = WORKDIR / "LJ_Valid_data"
TRANSCRIPTIONDIR = WORKDIR / "LJ_Valid_transcription"


class Whisper:
    """
    If ffmpeg gives error
    pip uninstall ffmpeg
    pip uninstall ffmpeg-python

    and install ffmpeg-python again with :
    pip install ffmpeg-python
    """

    def __init__(self, model="large"):
        print(f"[!] Loading model {model} ...")
        self.model = whisper.load_model(model)
        print("[+] Whisper model loaded")

    def transcribe(self, audio_path, lang="en"):
        result = self.model.transcribe(audio_path, language=lang)
        return result["text"]

    def transcribe_folder(self, input_dir, output_dir, exception=None, override=False):
        if exception:
            if isinstance(exception, list):
                exception = set(exception)
            if isinstance(exception, str):
                exception = {exception}

        print(f"[!] Running whisper over: {input_dir}")
        print(f"[!] Output will be saved in: {output_dir}")

        input_dir = Path(input_dir)
        output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        filelist = list(input_dir.rglob("*.wav"))
        pbar = tqdm(filelist)
        for filepath in pbar:
            in_filepath = list(filepath.parts)
            in_filepath[0] = output_dir
            output_path = Path(*in_filepath)
            if (
                exception and filepath.parent.name in exception
            ):  # change here the location to exclude
                print(f"\r[!] Skipping: {exception}")
                continue

            output_path.parent.mkdir(parents=True, exist_ok=True)
            transcription_path = output_path.with_suffix(".txt")
            if transcription_path.exists() and not override:
                pbar.write(f"\r[!] Skipping: {transcription_path} it already exists!")
                continue
            elif transcription_path.exists() and override:
                pbar.write(f"\r[!] Overriding: {transcription_path}")

            text = self.transcribe(str(filepath))

            with open(transcription_path, "w") as f:
                f.write(text)
        print(f"[+] Transcriptions saved to {output_dir}")


def main():
    model_choices = ["large", "medium.en", "small"]
    parser = argparse.ArgumentParser(description="Run whisper on a folder")
    parser.add_argument(
        "-m",
        "--model",
        default=model_choices[1],
        help="Model to use",
        choices=model_choices,
    )
    parser.add_argument("-i", "--input", help="Input folder", required=True)
    parser.add_argument("-o", "--output", help="Output folder", required=True)
    parser.add_argument(
        "-e", "--exceptions", nargs="+", help="Subfolders to exclude", default=None
    )
    parser.add_argument(
        "-f",
        "--override",
        help="Override existing transcriptions",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-v", "--verbose", help="Verbose mode", action="store_true", default=False
    )
    args = parser.parse_args()
    if args.verbose:
        print(args)

    whisper = Whisper(model=args.model)
    whisper.transcribe_folder(
        args.input, args.output, exception=args.exceptions, override=args.override
    )


if __name__ == "__main__":
    main()
