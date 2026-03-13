# Molecular Solubility Predictor (Biotech AI)

This project uses **PyTorch** and **RDKit** to predict the aqueous solubility of chemical compounds. 

### How it Works
1. **Data:** Uses the Delaney (ESOL) dataset.
2. **Featurization:** Converts SMILES strings into 2048-bit **Morgan Fingerprints** (ECFP4).
3. **Model:** A deep Feed-Forward Neural Network with Dropout for regularization.

### Performance
The model achieves a low Mean Squared Error (MSE) and can predict solubility (LogS) for unseen SMILES strings.

### Dependencies
* torch
* rdkit
* pandas
* scikit-learn
