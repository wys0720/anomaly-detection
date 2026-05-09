# SmoothNoiseAE

SmoothNoiseAE is a lightweight 1-D convolutional autoencoder for stationary noise reconstruction. It can be used as the reconstruction stage of a residual-based anomaly detection pipeline: the model learns background noise, and the residual between the input and reconstructed noise is later used for anomaly detection.

## Project structure

```text
SmoothNoiseAE/
├── train.py             # Main training entry
├── dataset.py           # SignalDataset for .txt/.npy signals
├── model.py             # SmoothNoiseAE model definition
├── engine.py            # train_one_epoch and validate
├── utils.py             # seed, plotting, checkpoint utilities
├── requirements.txt     # Python dependencies
├── README.md            # Usage instructions
└── .gitignore
```

## Installation

```bash
git clone <your-repository-url>
cd SmoothNoiseAE
pip install -r requirements.txt
```

If you want to train on GPU, install the PyTorch version that matches your CUDA environment.

## Data format

The training directory should contain 1-D signal samples saved as `.txt` or `.npy` files. Each file should contain one signal segment with length `4096` by default.

Example:

```text
data/noise/
├── noise_0001.txt
├── noise_0002.txt
├── noise_0003.npy
└── ...
```

Each sample is loaded as a tensor with shape `[1, length]`. By default, the signal is multiplied by `1e19`, following the original training script.

## Training

```bash
python train.py \
  --data-dir /path/to/noise/files \
  --output-dir outputs/smoothae \
  --epochs 20 \
  --batch-size 8 \
  --input-length 4096 \
  --scale 1e19
```

Windows example:

```bash
python train.py --data-dir "E:\\your_data\\noise" --output-dir "outputs\\smoothae"
```

If CPU training is unusually slow, you can limit PyTorch CPU threads:

```bash
python train.py --data-dir /path/to/noise/files --num-threads 1
```

## Outputs

After training, the output directory contains:

```text
outputs/smoothae/
├── best_model.pth       # Best validation checkpoint
├── loss_history.csv     # Train/validation loss values
└── loss_curve.png       # Loss curve figure
```

## Notes

- `SmoothNoiseAE` keeps the same core architecture as the original single-file `WhiteNoiseAE` training script.
- `WhiteNoiseAE = SmoothNoiseAE` is retained as an alias in `model.py` for backward compatibility.
- The data path is no longer hard-coded; use `--data-dir` to specify your training data.
