import json
from collections import Counter, defaultdict
from itertools import chain
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from whisperweranalysis.groundtruth import hvd_sentences
from whisperweranalysis.wer import compute


class MultipleSets:
    def __init__(self, transcription_loc, start=1, end=73) -> None:
        self.harvard_sets = []
        for set_no in range(start, end):
            hvd_set = HvdSet(set_no, hvd_sentences)
            hvd_set.compute_wer(transcription_loc)
            self.harvard_sets.append(hvd_set)

    def __getitem__(self, index):
        return self.harvard_sets[index - 1]

    def __iter__(self):
        return iter(self.harvard_sets)

    def __len__(self):
        return len(self.harvard_sets)

    def get_all_models_wer(self):
        assert len(self.harvard_sets) > 0, "No sets have been computed yet"
        models = defaultdict(lambda: defaultdict(float))
        for i, set_ in enumerate(self):
            for model in set_.wer:
                models[model][i + 1] = set_.wer[model]
        lens = [len(models[model]) for model in models]

        assert min(lens) == max(lens), "Not all models have the same number of sets"
        # Cast to dict from default dict and sorted by value
        return dict(
            {
                key: sorted(dict(value).items(), key=lambda x: x[1])
                for key, value in models.items()
            }
        )

    def get_top_n(self, n=10):
        models = self.get_all_models_wer()
        iterative_set = defaultdict(set)

        def get_set_intersection(defaultdict_):
            return set.intersection(*defaultdict_.values())

        # Get the intersection of the top n models for each set
        for i in range(len(models[list(models.keys())[0]])):
            # add one by one each element to the default dict of set
            for model in models:
                iterative_set[model].add(models[model][i][0])
            # Get intersections of all models
            final_set = get_set_intersection(iterative_set)
            # if the intersection is to the number required, return
            if len(final_set) >= n:
                break

        indices = sorted(list(final_set))
        return [self[i] for i in indices]


class HvdSet:
    def __init__(self, set_no, ground_truth):
        self.set_no = set_no
        self.begin = (set_no - 1) * 10 + 1
        self.end = (set_no) * 10
        self.transcriptions = defaultdict(list)
        self.wer = defaultdict(float)

        if isinstance(ground_truth, str):
            with open(ground_truth) as f:
                ground_truth = f.readlines()
        self.ground_truth = ground_truth[self.begin - 1 : self.end]

    def __repr__(self) -> str:
        return f"HvdSet({self.set_no}, {self.begin}, {self.end})"

    def get_text(self, dir):
        folders = [f for f in Path(dir).iterdir() if f.is_dir()]
        for folder in tqdm(folders, leave=False):
            for i in range(self.begin, self.end + 1):
                with open(f"{folder / folder.name}_{i}.txt") as f:
                    text = f.readline().strip()
                    self.transcriptions[folder.name].append(text if text != "" else ".")
        self.transcriptions = dict(self.transcriptions)

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
        for model, text in self.transcriptions.items():
            self.wer[model] = compute(self.ground_truth, text)
        self.wer = dict(self.wer)


class LJSpeech:
    def __init__(self, model_name, iterations="") -> None:
        self.model_name = model_name
        self.iterations = iterations
        self.transcriptions = []

    def __repr__(self) -> str:
        return f"LJSpeech({self.model_name}, {self.iterations}, len={len(self)})"

    def __len__(self):
        return len(self.transcriptions)

    def load_transcriptions(self, dir_path):
        if not isinstance(dir_path, Path):
            dir_path = Path(dir_path)
        dirs = sorted(list(dir_path.glob("*.txt")), key=lambda x: int(x.stem))
        for file in dirs:
            with open(file) as f:
                line = f.readline().strip()
                self.transcriptions.append(line if line else ".")

    def compute_wer(self, ground_truth):
        self.wer = compute(ground_truth, self.transcriptions)
        return self.wer

    @classmethod
    def load_from_dir(cls, dir_path):
        if not isinstance(dir_path, Path):
            dir_path = Path(dir_path)

        iter_name = dir_path.name
        model_name = dir_path.parent.name
        lj_class = cls(model_name, iter_name)
        lj_class.load_transcriptions(dir_path)
        return lj_class


class Predicted_MOS:
    def __init__(self, model_name, iterations, predicted_mos) -> None:
        self.model_name = model_name
        self.iterations = iterations
        self.predicted_mos = predicted_mos
        self.compute_stats()

    @classmethod
    def load_from_dir(cls, dir_path, filename="predicted_mos.json"):
        if not isinstance(dir_path, Path):
            dir_path = Path(dir_path)
        with open(dir_path / filename) as f:
            predicted_mos = json.load(f)
        iter_name = dir_path.name
        model_name = dir_path.parent.name
        return cls(model_name, iter_name, predicted_mos)

    def compute_stats(self):
        predicted_mos_values = list(self.predicted_mos.values())
        self.mean = np.mean(predicted_mos_values)
        self.std = np.std(predicted_mos_values)
        self.min = np.min(predicted_mos_values)
        self.max = np.max(predicted_mos_values)
        self.median = np.median(predicted_mos_values)
        self.mode = Counter(predicted_mos_values).most_common(1)[0][0]
        self.count = len(predicted_mos_values)

    def __repr__(self) -> str:
        return f"Predicted_MOS(model_name={self.model_name}, iterations={self.iterations}, mean={self.mean:.3f}, std={self.std:.3f}, min={self.min:.3f}, max={self.max:.3f}, median={self.median:.3f}, mode={self.mode:.3f}, count={self.count})"




class LenRTF:
    def __init__(self):
        self.storage = defaultdict(list)
        self.name = None
    
    def add(self, txt_len, rtf):
        self.storage[txt_len].append(rtf)
        
    @property
    def data(self):
        return dict(self.storage)
    
    def to_json(self, filename):
        json.dump(self.data, open(filename, "w"))
    
    @classmethod
    def from_json(cls, filename):
        obj = cls()
        obj.storage = defaultdict(list, json.load(open(filename)))
        obj.name = Path(filename).stem
        return obj

    def __repr__(self) -> str:
        array = np.array(list(chain.from_iterable(self.data.values())))
        return f"LenRTF({self.name}: len={len(self.data)}, mean±std={np.mean(array):.3f}±{np.std(array):.3f}, min:{np.min(array):.3f}, max={np.max(array):.3f})"

if __name__ == "__main__":
    # from whisperweranalysis.groundtruth import hvd_sentences
    # hvd = HvdSet(1, hvd_sentences)
    # hvd.compute_wer('whisperweranalysis/transcription')

    # from whisperweranalysis.structure import MultipleSets
    # hvd_sets = MultipleSets('/home/shivam/Projects/analysis-scripts/WhisperWERAnalysis/whisperweranalysis/transcription')
    # hvd_sets.get_top_n()

    # l = LJSpeech.load_from_dir('whisper-analysis-plots/whisperweranalysis/LJ_Valid_transcription/VOC/1000')
    # from whisperweranalysis.groundtruth import lj_valid
    # l.compute_wer(lj_valid)

    from whisperweranalysis.structure import MultipleSets

    hvd_sets = MultipleSets(
        "/home/smehta/Projects/whisper-analysis-plots/whisperweranalysis/transcription"
    )
