# analysis-utilities

Uses whisper default large to transcribe the text

## New

Now has command line scripts:

```
usage: resampler [-h] -i INPUT [-o OUTPUT] [-sr TARGET_SR]

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Input path to resample
  -o OUTPUT, --output OUTPUT
                        Output folder
  -sr TARGET_SR, --target-sr TARGET_SR
                        Target sampling rate
```

and

```
usage: whispertranscriber [-h] [-m {large,medium,small}] -i INPUT -o OUTPUT [-e EXCEPTIONS [EXCEPTIONS ...]]

Run whisper on a folder

optional arguments:
  -h, --help            show this help message and exit
  -m {large,medium,small}, --model {large,medium,small}
                        Model to use
  -i INPUT, --input INPUT
                        Input folder
  -o OUTPUT, --output OUTPUT
                        Output folder
  -e EXCEPTIONS [EXCEPTIONS ...], --exceptions EXCEPTIONS [EXCEPTIONS ...]
                        Subfolders to exclude
```

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
