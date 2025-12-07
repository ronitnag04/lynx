# Lynx ML Model

This folder contains the ML Model and associated scripts for the LYNX Neural stage. The model is based in PyTorch, and uses the AWS Neuron SDK to run on Trainium hardware.

### Contents

- **`model.py`**: Defines the `LynxMLModel` feed‑forward neural network (simple MLP with configurable hidden sizes).
- **`generate_dataset.py`**: Utility script to generate synthetic train/test datasets while developing the model and pipeline scripts.
- **`train.py`**: Trains `LynxMLModel` on the provided dataset using PyTorch with AWS Neuron.
- **`data/`** (created at runtime): Stores `train_features.npy`, `train_labels.npy`, `test_features.npy`, `test_labels.npy` produced by `generate_dataset.py`.
- **`checkpoints/`** (created at runtime): Stores model checkpoints (e.g. `checkpoint.pt`) saved by `train.py`.

### Requirements

- Trainium EC2 instance with the Neuron Deep Learning AMI. 

## Usage

### 0. Source the Neuron Environment
```bash
source /opt/aws_neuronx_venv_pytorch_2_8/bin/activate
```
Note that the exact path may be different depending on the AMI version you are using.

### 1. Generate synthetic data

```bash
python generate_dataset.py
```

This will create the `train_features.npy`, `train_labels.npy`, `test_features.npy`, and `test_labels.npy` files under `data/`.

### 2. Train the model

```bash
python train.py
```

Training logs and loss metrics will be printed to stdout, and a checkpoint will be written to `checkpoints/checkpoint.pt`.

## Notes 

### Clearing Neuron Compilation Cache
You may notice Neuron uses cached neffs to avoid recompilation:
```bash
INFO ||NEURON_CC_WRAPPER||: Using a cached neff at /var/tmp/neuron-compile-cache/neuronxcc-2.21.33363.0+82129205/MODULE_16389337710168549518+e30acd
3a/model.neff
```

In case you make changes to the files and Neuron uses the cached neffs instead of recompiling, you can clear the cached files with the following command:
```bash
rm -rf /var/tmp/neuron-compile-cache/
```
