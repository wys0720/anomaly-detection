# ADeepExtractor

ADeepExtractor (Advanced DeepExtractor) is a cross-window consistency noise reconstruction model for gravitational-wave time-series analysis. It extends a DeepExtractor-style STFT U-Net with adjacent-window joint modeling, bottleneck cross-window feature fusion, and attention-gated skip connections.

## Main idea

ADeepExtractor reconstructs detector background noise in the time--frequency domain. Each input window is converted into two channels:

1. normalized log-magnitude spectrogram;
2. normalized phase spectrogram.

During training, two adjacent windows are provided to the model. The two branches share the same encoder. Their bottleneck features are fused by a lightweight cross-window module, and the outputs are decoded into predicted noise spectra. The reconstructed noise is converted back to the time domain by iSTFT. The residual can then be obtained as:

```text
residual = input_signal - reconstructed_noise
```

## Repository structure

```text
ADeepExtractor_github/
├── model.py          # ADeepExtractor model modules
├── dataset.py        # Adjacent-window pair dataset with signal injection
├── stft_utils.py     # STFT/iSTFT encoding and decoding utilities
├── metrics.py        # PSNR, mismatch, PSD helpers, plotting utilities
├── train.py          # Training entry point
├── requirements.txt  # Python dependencies
├── .gitignore
└── README.md
```

## Data format

The current training script expects plain text files:

- `noise_dir/*.txt`: one 30 s noise sequence per file, length `122880` at 4096 Hz;
- `signal_dir/*.txt`: one 5 s signal sequence per file, length `20480` at 4096 Hz.

The script randomly injects a signal into a noise sequence and samples an adjacent-window pair from the mixed sequence. The corresponding clean noise windows are used as reconstruction targets.

Default lengths:

```text
Sampling rate       : 4096 Hz
Model window length : 28672 samples  (7 s)
Signal length       : 20480 samples  (5 s)
Noise sequence      : 122880 samples (30 s)
Pair stride         : 4096 samples   (1 s)
Overlap length      : 24576 samples  (6 s)
```

## Installation

Create an environment and install dependencies:

```bash
pip install -r requirements.txt
```

Install PyTorch according to your CUDA version if needed. See the official PyTorch installation instructions for GPU-specific commands.

## Training

Example:

```bash
python train.py \
  --signal_dir /path/to/Signal_4096 \
  --noise_dir /path/to/noise_4096_train \
  --output_dir outputs/adeepextractor_snr15 \
  --epochs 50 \
  --batch_size 8 \
  --lr 3e-4 \
  --noise_scale 80.0 \
  --signal_scale 0.83 \
  --p_signal_pair 0.7
```

Outputs saved in `output_dir`:

- `best_model_adeepextractor.pth`: best model checkpoint;
- `norm_params_adeepextractor.json`: normalization and STFT parameters needed for inference;
- `train_config.json`: training configuration;
- `loss_curves_cross_window_pair.png`: training loss curves;
- diagnostic figures for spectrogram and waveform reconstruction.

## Loss functions

The training objective combines four terms:

```text
L = lambda_mag * L_mag
  + lambda_phase * L_phase
  + lambda_time * L_time
  + lambda_overlap * L_overlap
```

where:

- `L_mag`: MSE loss on normalized log-magnitude predictions;
- `L_phase`: phase consistency loss based on cosine phase difference;
- `L_time`: L1 loss between reconstructed and target noise in the time domain;
- `L_overlap`: consistency loss between the overlapping region of the two adjacent window outputs.

## Notes

- The code is intentionally modular so that `model.py` can be reused in inference scripts.
- If your training data uses different lengths or scaling factors, update the command-line arguments or the defaults in `TrainConfig`.
- Large data files and model checkpoints are ignored by `.gitignore` and should not be committed to GitHub.
