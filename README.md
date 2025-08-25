# torch-cellfm

This project is a PyTorch reimplementation and migration of the original CellFM model, supporting loading of the original 80M MindSpore pre-trained weights for downstream tasks.

## Project Overview

 `torch-cellfm` provides a faithful PyTorch version of CellFM, making it easier for the community to train and deploy the model using the mainstream PyTorch framework.

**Key Features:**

- Complete reimplementation of the CellFM architecture and training pipeline in PyTorch
- Support for loading original MindSpore 80M pre-trained weights
- Ready for downstream fine-tuning and custom task extension
- Clean and modular codebase for easy development

## Installation
```
conda create -n CellFM_torch python=3.9
conda activate CellFM_torch
```
and then install the required packages below:

- mindspore
- scanpy
- scib

- torch
- numpy
- pandas
- tqdm

## Quick Start

### Data Preprocessing
The data preprocessing workflow is identical to the original CellFM implementation. Please follow the same steps as described in the original CellFM documentation to prepare your datasets.

### Training or Fine-tuning on a New Dataset

We provide the `main.py` script for fine-tuning or training CellFM on new datasets. Below is an example command to train on the `Pancrm0` dataset using a single GPU:

```
python main.py --dataset Pancrm0 --batch_size 16 --device cuda:2 --epoch 5 --ckpt_path "/bigdat2/user/shanggny/
checkpoint/para80m/6300w_18000_19479-1_38071.ckpt" --feature_col cell_type
```

- dataset: Name of the dataset to load. Split into train.h5ad and test.h5ad.
- batch_size: Number of samples per training step.
- device: Compute device to run on. Use cpu or cuda:<gpu_id> (e.g., cuda:0, cuda:2).

- epoch: Number of full passes over the training dataset.
- ckpt_path: Path to a pre-train model weights (mindspore weight).
- feature_col: Column name in adata.obs used as target/label (e.g., cell_type, batch). Determines the supervised task target.
