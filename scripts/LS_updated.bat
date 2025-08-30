@echo off
REM === Go to repo root (one level up from scripts folder) ===
cd ..

REM === LibriSpeech + 0 (no noise) ===
python main.py --asr facebook/wav2vec2-base-960h --steps 10 --dataset_name librispeech --dataset_dir LibriSpeech --temp 2.5 --episodic --em_coef 0.3 --reweight --log_dir exps --lr 2e-5 --non_blank --train_feature --extra_noise 0

REM === LibriSpeech + 0.005 noise ===
python main.py --asr facebook/wav2vec2-base-960h --steps 10 --dataset_name librispeech --dataset_dir LibriSpeech --temp 2.5 --episodic --em_coef 0.3 --reweight --log_dir exps --lr 2e-5 --non_blank --train_feature --extra_noise 0.005

REM === LibriSpeech + 0.01 noise ===
python main.py --asr facebook/wav2vec2-base-960h --steps 10 --dataset_name librispeech --dataset_dir LibriSpeech --temp 2.5 --episodic --em_coef 0.3 --reweight --log_dir exps --lr 2e-5 --non_blank --train_feature --extra_noise 0.01
