# analysis-utilities

Uses whisper default large to transcribe the text

## Usage

1. Use `transcriber.Whisper` to transcribe the the folder of audio files

```python
whisper = Whisper()
whisper.transcribe_folder(DATADIR)
```

2. Use `wer.compute` to compute the wer

```python
compute(reference, transcription)
```

3. For harvard sentences and folder strcuture like

```
- data
    |- model1
        |- model1_1.txt
        |- model1_2.txt
    |- model2
        |- model2_1.txt
        |- model2_2.txt
```

Use `structure.HvdSet`

```python
from groundtruth import HvdSet
hvdset = HvdSet(set_no=1, ground_truth=hvd_sentences)
```




Use checkpoint_mover to move the iteration checkpoint to a sub folder

```python
list_iters = get_list_iter()
print(len(list_iters))
print(list_iters)
copy_checkpoints("./logs/blank", "glow_checkpoints", list_iters)
```