from collections import defaultdict
from pathlib import Path

from tqdm.auto import tqdm

from wer import compute


class HvdSet:
    def __init__(self, set_no, ground_truth):
        self.set_no = set_no
        self.begin = set_no * 10 + 1
        self.end = (set_no + 1) * 10
        self.models = defaultdict(list)
        self.wer = defaultdict(float)
        
        if isinstance(ground_truth, str):
            with open(ground_truth) as f:
                ground_truth = f.readlines()
        self.ground_truth = ground_truth[self.begin - 1: self.end]

    def get_text(self, dir):
        folders = [f for f in Path(dir).iterdir() if f.is_dir()]
        for folder in tqdm(folders, leave=False):
            for i in range(self.begin, self.end + 1):
                with open(f"{folder}_{i}.txt") as f:
                    self.model[folder].append(f.readline())

    def compute_wer(self, dir):
        """
        Expects the folder strcutre to be:
        - data
            |- model1
                |- model1_1.txt
                |- model1_2.txt
            |- model2
                |- model2_1.txt
                |- model2_2.txt
        ...

        Args:
            dir (_type_): _description_
        """
        self.get_text(dir)
        for model, text in self.models.items():
            self.wer[model] = compute(self.ground_truth, text)
            




