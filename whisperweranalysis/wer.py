import argparse
from pathlib import Path
from typing import List

import jiwer


class DisfulencyRemover(jiwer.transforms.AbstractTransform):
    def __init__(self) -> None:
        super().__init__()
        self.disfluencies = {"uh", "uhm", "ah", "uck"}

    def process_string(self, s: List):
        # import pdb; pdb.set_trace()
        return [x for x in s if x not in self.disfluencies]

    def process_list(self, inp: List[str]):
        return [self.process_string(s) for s in inp]


normalising_transformation = jiwer.Compose(
    [
        jiwer.ToLowerCase(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.RemoveEmptyStrings(),
        jiwer.RemovePunctuation(),
        jiwer.ExpandCommonEnglishContractions(),
        jiwer.ReduceToListOfListOfWords(),
        DisfulencyRemover(),
    ]
)


def compute(reference, hypothesis):
    """
    Calculates the Word Error Rate (WER) between two lists of transcriptions

    Args:
        reference(List[str]): list of references
        transcription(List[str]): list of transcriptions

    Returns:
        float: WER between transcriptions and references
    """
    return (
        jiwer.wer(
            reference,
            hypothesis,
            truth_transform=normalising_transformation,
            hypothesis_transform=normalising_transformation,
        )
        * 100
    )


def load_transcriptions_from_folder(folder_path):
    """
    Load transcriptions from a folder

    Args:
        folder_path(str): path to folder containing transcriptions

    Returns:
        List
    """
    if isinstance(folder_path, str):
        folder_path = Path(folder_path)

    all_transcriptions = []
    for file in folder_path.glob("*.txt"):
        with open(file) as f:
            all_transcriptions.append(f.read().strip())

    return all_transcriptions


def main():
    parser = argparse.ArgumentParser(description="Calculate WER between two folders")

    parser.add_argument("-r", "--reference", help="Input ground truth", required=True)
    parser.add_argument("-i", "--input", help="Input folder", required=True)
    parser.add_argument(
        "-rt", "--reference_type", choices=["folder", "file"], default="folder"
    )
    parser.add_argument(
        "-it", "--input_type", choices=["folder", "file"], default="folder"
    )
    parser.add_argument(
        "-v", "--verbose", help="Verbose mode", action="store_true", default=False
    )
    args = parser.parse_args()
    if args.verbose:
        print(args)

    if args.reference_type == "folder":
        reference = load_transcriptions_from_folder(args.reference)
    else:
        with open(args.reference) as f:
            reference = f.readlines()

    if args.input_type == "folder":
        hypothesis = load_transcriptions_from_folder(args.input)
    else:
        with open(args.input) as f:
            hypothesis = f.readlines()

    wer = compute(reference, hypothesis)
    print(f"WER: {wer:.4f}%")


if __name__ == "__main__":
    main()
