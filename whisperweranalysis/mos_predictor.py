# ==============================================================================
# Copyright (c) 2021, Yamagishi Laboratory, National Institute of Informatics
# Author: Erica Cooper
# All rights reserved.
# ==============================================================================

import argparse
import json
import os
import random

import fairseq
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio

# import soundfile as sf
import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

random.seed(1984)


class MosPredictor(nn.Module):
    def __init__(self, ssl_model, ssl_out_dim):
        super().__init__()
        self.ssl_model = ssl_model
        self.ssl_features = ssl_out_dim
        self.output_layer = nn.Linear(self.ssl_features, 1)

    def forward(self, wav):
        wav = wav.squeeze(1)  # [batches, audio_len]
        res = self.ssl_model(wav, mask=False, features_only=True)
        x = res["x"]
        x = torch.mean(x, 1)
        x = self.output_layer(x)
        return x.squeeze(1)


class MyDataset(Dataset):
    def __init__(self, wavdir, mos_list):
        self.mos_lookup = {}
        f = open(mos_list)
        for line in f:
            parts = line.strip().split(",")
            wavname = parts[0]
            mos = float(parts[1])
            self.mos_lookup[wavname] = mos

        self.wavdir = wavdir
        self.wavnames = sorted(self.mos_lookup.keys())

    def __getitem__(self, idx):
        wavname = self.wavnames[idx]
        wavpath = os.path.join(self.wavdir, wavname)
        wav = torchaudio.load(wavpath)[0]
        score = self.mos_lookup[wavname]
        return wav, score, wavname

    def __len__(self):
        return len(self.wavnames)

    def collate_fn(self, batch):  # zero padding
        wavs, scores, wavnames = zip(*batch)
        wavs = list(wavs)
        max_len = max(wavs, key=lambda x: x.shape[1]).shape[1]
        output_wavs = []
        for wav in wavs:
            amount_to_pad = max_len - wav.shape[1]
            padded_wav = torch.nn.functional.pad(wav, (0, amount_to_pad), "constant", 0)
            output_wavs.append(padded_wav)

        output_wavs = torch.stack(output_wavs, dim=0)
        scores = torch.stack([torch.tensor(x) for x in list(scores)], dim=0)
        return output_wavs, scores, wavnames


def train():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datadir", type=str, required=True, help="Path of your DATA/ directory"
    )
    parser.add_argument(
        "--fairseq_base_model",
        type=str,
        required=True,
        help="Path to pretrained fairseq base model",
    )
    parser.add_argument(
        "--finetune_from_checkpoint",
        type=str,
        required=False,
        help="Path to your checkpoint to finetune from",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        required=False,
        default="checkpoints",
        help="Output directory for your trained checkpoints",
    )
    args = parser.parse_args()

    cp_path = args.fairseq_base_model
    datadir = args.datadir
    ckptdir = args.outdir
    my_checkpoint = args.finetune_from_checkpoint

    if not os.path.exists(ckptdir):
        os.system("mkdir -p " + ckptdir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("DEVICE: " + str(device))

    wavdir = os.path.join(datadir, "wav")
    trainlist = os.path.join(datadir, "sets/train_mos_list.txt")
    validlist = os.path.join(datadir, "sets/val_mos_list.txt")

    ssl_model_type = cp_path.split("/")[-1]
    if ssl_model_type == "wav2vec_small.pt":
        SSL_OUT_DIM = 768
    elif ssl_model_type in ["w2v_large_lv_fsh_swbd_cv.pt", "xlsr_53_56k.pt"]:
        SSL_OUT_DIM = 1024
    else:
        print("*** ERROR *** SSL model type " + ssl_model_type + " not supported.")
        exit()

    model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
    ssl_model = model[0]
    ssl_model.remove_pretraining_modules()

    trainset = MyDataset(wavdir, trainlist)
    trainloader = DataLoader(
        trainset,
        batch_size=4,
        shuffle=True,
        num_workers=2,
        collate_fn=trainset.collate_fn,
    )

    validset = MyDataset(wavdir, validlist)
    validloader = DataLoader(
        validset,
        batch_size=2,
        shuffle=True,
        num_workers=2,
        collate_fn=validset.collate_fn,
    )

    net = MosPredictor(ssl_model, SSL_OUT_DIM)
    net = net.to(device)

    if my_checkpoint is not None:  # do (further) finetuning
        net.load_state_dict(torch.load(my_checkpoint))

    criterion = nn.L1Loss()
    optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)

    PREV_VAL_LOSS = 9999999999
    orig_patience = 20
    patience = orig_patience
    for epoch in range(1, 1001):
        STEPS = 0
        net.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels, filenames = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            STEPS += 1
            running_loss += loss.item()
        print("EPOCH: " + str(epoch))
        print("AVG EPOCH TRAIN LOSS: " + str(running_loss / STEPS))
        epoch_val_loss = 0.0
        net.eval()
        # clear memory to avoid OOM
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
        # validation
        VALSTEPS = 0
        for i, data in enumerate(validloader, 0):
            VALSTEPS += 1
            inputs, labels, filenames = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            epoch_val_loss += loss.item()

        avg_val_loss = epoch_val_loss / VALSTEPS
        print("EPOCH VAL LOSS: " + str(avg_val_loss))
        if avg_val_loss < PREV_VAL_LOSS:
            print("Loss has decreased")
            PREV_VAL_LOSS = avg_val_loss
            PATH = os.path.join(ckptdir, "ckpt_" + str(epoch))
            torch.save(net.state_dict(), PATH)
            patience = orig_patience
        else:
            patience -= 1
            if patience == 0:
                print(
                    "loss has not decreased for "
                    + str(orig_patience)
                    + " epochs; early stopping at epoch "
                    + str(epoch)
                )
                break

    print("Finished Training")


# ==============================================================================
# Adapted from predict.py mos-finetune-ssl by Erica Cooper
# ==============================================================================


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fairseq-base-model",
        type=str,
        default="/home/smehta/Projects/whisper-analysis-plots/wav2vec_small.pt",
        help="Path to pretrained fairseq base model.",
    )
    # parser.add_argument('--datadir', type=str, required=True, help='Path of your DATA/ directory')
    parser.add_argument(
        "--finetuned-checkpoint",
        type=str,
        default="/home/smehta/Projects/whisper-analysis-plots/ckpt_w2vsmall",
        help="Path to finetuned MOS prediction checkpoint.",
    )
    # parser.add_argument('--outfile', type=str, required=False, default='answer.txt',
    # help='Output filename for your answer.txt file for submission to the CodaLab leaderboard.')
    parser.add_argument(
        "--device",
        type=str,
        required=False,
        default="cuda:0",
        help="Device to use for inference.",
    )
    parser.add_argument(
        "--wav-fpath",
        type=str,
        default=None,
        help="Path to wav file to predict MOS for.",
    )
    parser.add_argument(
        "-i",
        "--wav-dir",
        type=str,
        default=None,
        help="Path to directory containing wav files to predict MOS for.",
    )
    args = parser.parse_args()
    assert (
        args.wav_fpath is not None or args.wav_dir is not None
    ), "Either wav_fpath or wav_dir must be specified."
    return args


def get_mos_model(cp_path, my_checkpoint, device):
    # datadir = args.datadir
    # outfile = args.outfile

    # system_csv_path = os.path.join(datadir, 'mydata_system.csv')

    model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
    ssl_model = model[0]
    ssl_model.remove_pretraining_modules()

    print("Loading checkpoint")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    ssl_model_type = cp_path.split("/")[-1]
    if ssl_model_type == "wav2vec_small.pt":
        SSL_OUT_DIM = 768
    elif ssl_model_type in ["w2v_large_lv_fsh_swbd_cv.pt", "xlsr_53_56k.pt"]:
        SSL_OUT_DIM = 1024
    else:
        print("*** ERROR *** SSL model type " + ssl_model_type + " not supported.")
        exit()

    model = MosPredictor(ssl_model, SSL_OUT_DIM).to(device)
    model.eval()

    model.load_state_dict(torch.load(my_checkpoint))

    return model


def load_wav(wav_fpath, device, resample_tol=1e-1):
    wav, sr = torchaudio.load(wav_fpath)
    if sr != 16000:
        # print('Resampling from {} to 16000'.format(sr))
        resampler = torchaudio.transforms.Resample(sr, 16000)
        wav = resampler(wav)
    # assert sr == 16000, "Sample rate mismatch: {} vs 16000".format(sr)
    assert (
        torch.max(torch.abs(wav)) - resample_tol <= 1.0
    ), "wav file should be normalized to [-1, 1]"
    wav[torch.max(wav) > 1.0] = 1.0
    wav[torch.min(wav) < -1.0] = -1.0
    wav = wav.to(device)
    wav = wav.unsqueeze(0)
    return wav


def predict_mos(wav_fpath, model, device):
    wav = load_wav(wav_fpath, device)

    with torch.no_grad():
        mos = model(wav)
        mos = mos.cpu().numpy()
    return mos[0]


def predict_mos_dir(wav_dir, model, device, save_in_dir=True):
    mos_dict = {}
    print("Predicting MOS for files in", wav_dir)
    wav_fpaths = [
        os.path.join(wav_dir, f) for f in os.listdir(wav_dir) if f.endswith(".wav")
    ]
    for wav_fpath in tqdm.tqdm(wav_fpaths):
        wav = load_wav(wav_fpath, device)
        wav_fname = os.path.basename(wav_fpath)
        with torch.no_grad():
            mos_dict[wav_fname] = float(model(wav).cpu().numpy()[0])
    mos = np.array(list(mos_dict.values()))
    print("mean mos:", np.mean(mos))
    print("std mos:", np.std(mos))
    print("total audio samples:", len(mos))

    summary = {
        "mean_mos": float(np.mean(mos)),
        "std_mos": float(np.std(mos)),
        "total_audio_samples": int(len(mos)),
    }

    if save_in_dir:
        json.dump(mos_dict, open(os.path.join(wav_dir, "predicted_mos.json"), "w"))
        json.dump(summary, open(os.path.join(wav_dir, "mos_summary.json"), "w"))
    return mos


def main():
    # Exposed to script
    args = parse_args()
    model = get_mos_model(
        cp_path=args.fairseq_base_model,
        my_checkpoint=args.finetuned_checkpoint,
        device=args.device,
    )
    device = torch.device(args.device)

    if args.wav_fpath is not None:
        mos = predict_mos(args.wav_fpath, model, device)
        print("predicted mos:", mos)
    elif args.wav_dir is not None:
        mos = predict_mos_dir(args.wav_dir, model, device)
        # print("predicted mos:", mos)


if __name__ == "__main__":
    main()
