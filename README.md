# Kaggle-HuBMAP
Codes for Kaggle competition: HuBMAP - Hacking the Kidney

# Instructions

## Environment

All codes are based on `pytorch` and `pytorch-lightning`, and other common deep learning, image processing packages. Most of imported packages can be easily installed by conda or pip.

## Data processing

Training data abd public test data are from Kaggle competition data source, and external data is downloaded from official HuBMAP dataset port.

- `Visual.ipynb` is for visualization of images and corresponding mask, which is not neccessary.

- `Slicer.ipynb` is for slicing WSI into smaller patches. This should be executed before training.

## Training phase

This code version only contains EfficinetNet-b3-U-Net training. For other backbones, please refer to `timm` packages and `segmentation_models_pytorch` for available U-Net backbones.

- `Solver.py` is training code of training phase without pseudo-labeling. 

- Simply run `python Solver.py`.

- Trianing config should be set inside `Solver.py`, including learning rate, batch size, number of epochs, backbone and so on. Most of configs are shown in first several lines of `Solver.py`

- `SolverPseudo.py` is training code of training phase **with** pseudo-labeling. 

## Inferencing phase

Run `Predictor.py` or `NewPredictor.py` to inference on datasets. 
