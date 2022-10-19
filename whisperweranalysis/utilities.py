import logging
from collections import defaultdict
from typing import Dict, List

import numpy as np

from whisperweranalysis.structure import HvdSet, MultipleSets

logger = logging.getLogger(__name__)


def get_all_values(list_wer_dicts : List[Dict]) -> Dict:
    assert len(list_wer_dicts) > 0, "List of wer dicts is empty"
    if isinstance(list_wer_dicts[0], (HvdSet, MultipleSets)):
        logger.info("Converting HvdSet to dict")
        list_wer_dicts = [hvd.wer for hvd in list_wer_dicts]

    accumulated = defaultdict(list)
    for wer_dict in list_wer_dicts:
        for model in wer_dict:
            accumulated[model].append(wer_dict[model])

    return dict(accumulated)

def get_statistics_of_wer(list_wer_dicts : List[Dict]):
    accumulated = get_all_values(list_wer_dicts)
    return {model : np.mean(accumulated[model]) for model in accumulated}, {model : np.std(accumulated[model]) for model in accumulated}



## TODO:
### Copy GT code
def copy_gt():
    import shutil
    from pathlib import Path

    from tqdm.auto import tqdm

    filelist = Path('/hdd/shivam/Project/Neural-FHMM/data/filelists/ljs_audio_text_val_filelist.txt')
    wav_loc = Path('/hdd/shivam/Project/Neural-FHMM')
    SAVE_LOC = Path('whisperweranalysis/LJ_Valid_data/GT')

    with open(filelist, 'r') as f:
        for i, line in enumerate(tqdm(f.readlines())):
            sub_loc = line.strip().split('|')[0]
            wav_path = wav_loc / sub_loc
            shutil.copy(wav_path, SAVE_LOC / f"{i + 1}.wav")
