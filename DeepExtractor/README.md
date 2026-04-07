# Gravitational-Wave Noise Reconstruction (STFT + U-Net)

This repository contains a cleaned project skeleton for STFT-domain noise reconstruction using a 2D U-Net. It is adapted from a single research script and reorganized for public release.

## Project structure

```text
.
├── config.py
├── dataset.py
├── infer.py
├── models.py
├── signal_utils.py
├── train.py
├── requirements.txt
└── README.md
```

## What each file does

- `config.py`: shared runtime parameters and default paths
- `dataset.py`: dataset definition for signal+noise mixtures and target noise
- `models.py`: model architectures
- `signal_utils.py`: STFT/iSTFT, PSD, mismatch, JSON helpers
- `visualize.py`: plotting helpers
- `train.py`: training entry point
- `infer.py`: inference entry point

## Expected data layout

Create your own directories and update `PathConfig` in `config.py`:

```text
./data/
├── signals/
│   ├── sample_001.txt
│   └── ...
├── noise_train/
│   ├── noise_001.txt
│   └── ...
├── test_signal.txt
└── test_noise.txt
```

## Install

```bash
pip install -r requirements.txt
```

## Train

```bash
python train.py
```

## Inference

```bash
python infer.py
```

## Notes

- This skeleton keeps the original training logic as much as possible, but removes large blocks of deprecated/commented code.
- Before publishing, you should replace local absolute paths with your own dataset layout.
- Add a `LICENSE` file before making the repository public.
