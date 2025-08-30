#!/bin/bash

# === LibriSpeech + 0 (no noise) ===
python ../main.py \
  --asr facebook/wav2vec2-base-960h \
  --steps 10 \
  --dataset_name librispeech \
  --dataset_dir ../LibriSpeech \
  --temp 2.5 \
  --episodic \
  --em_coef 0.3 \
  --reweight \
  --log_dir ../exps/noise_0 \
  --lr 2e-5 \
  --non_blank \
  --train_feature \
  --extra_noise 0

# === LibriSpeech + 0.005 noise ===
python ../main.py \
  --asr facebook/wav2vec2-base-960h \
  --steps 10 \
  --dataset_name librispeech \
  --dataset_dir ../LibriSpeech \
  --temp 2.5 \
  --episodic \
  --em_coef 0.3 \
  --reweight \
  --log_dir ../exps/noise_0.005 \
  --lr 2e-5 \
  --non_blank \
  --train_feature \
  --extra_noise 0.005

# === LibriSpeech + 0.01 noise ===
python ../main.py \
  --asr facebook/wav2vec2-base-960h \
  --steps 10 \
  --dataset_name librispeech \
  --dataset_dir ../LibriSpeech \
  --temp 2.5 \
  --episodic \
  --em_coef 0.3 \
  --reweight \
  --log_dir ../exps/noise_0.01 \
  --lr 2e-5 \
  --non_blank \
  --train_feature \
  --extra_noise 0.01
