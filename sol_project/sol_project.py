import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class MoleculePredictor(nn.Module):
    def __init__(self, input_size):
        super(MoleculePredictor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.network(x)


def smiles_to_fp(smiles, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits)
        return np.array(fp)
    return np.zeros(n_bits)


class MoleculeDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]


def run_biotech_pipeline():
    # Get Data (Delaney ESOL)
    url = "https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/delaney-processed.csv"
    df = pd.read_csv(url)
    print(f"Dataset loaded: {len(df)} molecules found.")

    # Featurize
    print("Converting molecules to Morgan Fingerprints")
    X = np.array([smiles_to_fp(s) for s in df['smiles']])
    y = df['measured log solubility in mols per litre'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    y_train_scaled = scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = scaler.transform(y_test.reshape(-1, 1)).flatten()

    train_loader = DataLoader(MoleculeDataset(X_train, y_train_scaled), batch_size=32, shuffle=True)
    test_loader = DataLoader(MoleculeDataset(X_test, y_test_scaled), batch_size=32)

    model = MoleculePredictor(input_size=2048)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("\nStarting Training")
    for epoch in range(20):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1} | Avg Loss: {total_loss/len(train_loader):.4f}")

    # on a new molecule
    model.eval()
    test_smiles = "c1ccccc1"  # Benzene
    with torch.no_grad():
        fp = torch.tensor(smiles_to_fp(test_smiles), dtype=torch.float32).unsqueeze(0)
        scaled_pred = model(fp).item()
        actual_pred = scaler.inverse_transform([[scaled_pred]])[0][0]

    print("-" * 30)
    print(f"Prediction for {test_smiles} (Benzene):")
    print(f"Predicted Solubility (LogS): {actual_pred:.2f}")
    from sklearn.metrics import mean_squared_error

    # model.eval()
    with torch.no_grad():
        test_preds_scaled = model(torch.tensor(X_test, dtype=torch.float32)).numpy().flatten()
    test_preds = scaler.inverse_transform(test_preds_scaled.reshape(-1, 1)).flatten()
    rmse = np.sqrt(mean_squared_error(y_test, test_preds))
    print(f"Held-out Test RMSE (LogS): {rmse:.3f}")


if __name__ == "__main__":
    run_biotech_pipeline()
