import argparse

import jiwer

normalising_transformation = jiwer.Compose(
    [
        jiwer.ToLowerCase(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.RemoveEmptyStrings(),
        jiwer.RemovePunctuation(),
        jiwer.ExpandCommonEnglishContractions(),
        jiwer.ReduceToListOfListOfWords(),
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
    all_transcriptions = []
    for file in folder_path.glob("*.txt"):
        with open(file) as f:
            all_transcriptions.append(f.read().strip())

    return all_transcriptions


def main():
    parser = argparse.ArgumentParser(description="Calculate WER between two folders")

    parser.add_argument("-r", "--reference", help="Input ground truth", required=True)
    parser.add_argument("-i", "--input", help="Input folder", required=True)
    args = parser.parse_args()
    print(args)

    reference = load_transcriptions_from_folder(args.reference)
    hypothesis = load_transcriptions_from_folder(args.input)
    wer = compute(reference, hypothesis)
    print(f"WER: {wer:.4f}%")


if __name__ == "__main__":
    main()
