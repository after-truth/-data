# EEG Decoding Project

## Quick Start
python3 -m pip install --break-system-packages -r requirements.txt
python3 scripts/train.py --config configs/train_cross_subject.yaml
python3 scripts/train.py --config configs/train_cross_subject_ssm.yaml
python3 scripts/train.py --config configs/train_cross_subject_mlp_ssm.yaml
