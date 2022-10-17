import jiwer

normalising_transformation = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.RemoveEmptyStrings(),
    jiwer.RemovePunctuation(),
    jiwer.ReduceToListOfListOfWords()
])



def compute(reference, hypothesis):
    """
    Calculates the Word Error Rate (WER) between two lists of transcriptions

    Args:
        reference(List[str]): list of references
        transcription(List[str]): list of transcriptions

    Returns:
        float: WER between transcriptions and references
    """
    return jiwer.wer(reference, hypothesis, truth_transform=normalising_transformation,
                   hypothesis_transform=normalising_transformation)
