from src.preprocessing.data_preprocessing import preprocess_data

if __name__ == "__main__":
    preprocess_data(
        "data/chem/raw/SMILES_TrainingDataset_July2_172_Plus_16LY_Fixed.xlsx",
        "data/chem/raw/SMILES_TestingDataset_July3_Final.xlsx"
    )