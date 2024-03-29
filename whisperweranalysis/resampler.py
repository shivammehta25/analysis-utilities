#!/usr/bin/env python
import argparse
from pathlib import Path

import librosa
import soundfile as sf
from tqdm.auto import tqdm


def resampleit(path, output, target_sr, extension):
    current_path = Path(path)
    total_files = list(current_path.glob(f"*.{extension}"))
    print(f"[!] Found: {len(total_files)} files at {current_path}")
    output_path = current_path / output
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"[!] Saving to {output_path}")
    for f_name in tqdm(total_files):
        if f".{extension}" in str(f_name):
            y, sr = librosa.load(f_name)
            y_ = librosa.resample(y, orig_sr=sr, target_sr=target_sr)

            sf.write(output_path / f"{f_name.stem}.wav", y_, target_sr)
    print("[+] Resampled!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="Input path to resample", required=True)
    parser.add_argument("-o", "--output", help="Output folder", default="out")
    parser.add_argument(
        "-sr", "--target-sr", help="Target sampling rate", default=22050, type=int
    )
    parser.add_argument(
        "-e", "--extension", help="Glob pattern to search for", default="wav"
    )
    args = parser.parse_args()
    print(args)
    resampleit(args.input, args.output, args.target_sr, args.extension)


if __name__ == "__main__":
    main()
