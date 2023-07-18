# analysis-utilities

Uses whisper default large to transcribe the text

## New

Now has command line scripts:

1. Resample

```bash
usage: resampler [-h] -i INPUT [-o OUTPUT] [-sr TARGET_SR] [-e EXTENSION]

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Input path to resample
  -o OUTPUT, --output OUTPUT
                        Output folder
  -sr TARGET_SR, --target-sr TARGET_SR
                        Target sampling rate
  -e EXTENSION, --extension EXTENSION
                        Glob pattern to search for
```

2. Transcribe with whisper

```bash
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

3. Collate the transcriptions

```bash
usage: collatetranscription [-h] -p PATH [-o OUTPUT]

optional arguments:
  -h, --help            show this help message and exit
  -p PATH, --path PATH  Path to transcriptions
  -o OUTPUT, --output OUTPUT
                        Path to output file
```

4. Audio 2 Mel: using old STFT implementation not torch's but tacotron 2s

```bash
usage: audio2mel [-h] -i INPUT [-o OUTPUT] [-sr SAMPLING_RATE]

Convert folder to mels

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Input folder
  -o OUTPUT, --output OUTPUT
                        Output folder
  -sr SAMPLING_RATE, --sampling_rate SAMPLING_RATE
                        Target sampling rate
```

5. Delete my checkpoints

```bash
Usage: checkpointdeleter [OPTIONS]

Options:
  -i, --dir_path PATH  Path to checkpoint directory where checkpoints will be
                       filtered  [required]
  -e, --ext TEXT       Extension of checkpoints to filter
  --help               Show this message and exit.
```

6. Predict MOS scores

```bash
usage: predict_mos [-h] [--fairseq-base-model FAIRSEQ_BASE_MODEL] [--finetuned-checkpoint FINETUNED_CHECKPOINT] [--device DEVICE]
                   [--wav-fpath WAV_FPATH] [--wav-dir WAV_DIR]

optional arguments:
  -h, --help            show this help message and exit
  --fairseq-base-model FAIRSEQ_BASE_MODEL
                        Path to pretrained fairseq base model.
  --finetuned-checkpoint FINETUNED_CHECKPOINT
                        Path to finetuned MOS prediction checkpoint.
  --device DEVICE       Device to use for inference.
  --wav-fpath WAV_FPATH
                        Path to wav file to predict MOS for.
  --wav-dir WAV_DIR     Path to directory containing wav files to predict MOS for.
```

## Available data structures

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
