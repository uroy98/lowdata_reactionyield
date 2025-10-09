#short overview of project, dataset info, environment setup, how to run scripts ( Tentative )

# Low-Data Reaction Yield Prediction

This repository contains code and datasets for our low-data machine learning models for predicting reaction yields in micellar catalysis.

## Folder structure
- `src/preprocessing/` — SMILES cleaning and feature generation  
- `src/models/` — Hybrid and baseline models  
- `src/experiments/` — Training scripts  
- `data/chem/` — Datasets (raw Excel files, processed .npy)  
- `results/` — Generated metrics, predictions, and plots  

## How to run
```bash
conda create -n lowdata_reactionyield python=3.10 -y
conda activate lowdata_reactionyield
pip install -r environment/requirements.txt
python run_preprocessing.py
python run_baselines_custom_preprocessing.py
python run_representation_baselines.py
