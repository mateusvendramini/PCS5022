import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import lightning as L
from torch.utils.data import DataLoader, TensorDataset
import os

# 1. Carregar dataset
# Usar o mesmo caminho do kaggle_02.py
DATA_PATH = 'kaggle/input/pcs-5022-competicao-deca-learn-202-task-2/dataset2.npz'
data = np.load(DATA_PATH)
X_train = data['X_train']
X_train = np.nan_to_num(X_train, nan=0.0)
y_train = data['y_train']
X_val = data['X_val']
X_val = np.nan_to_num(X_val, nan=0.0)
y_val = data['y_val']
X_test = data['X_test']
X_test = np.nan_to_num(X_test, nan=0.0)

# 2. Normalização
mean = X_train.mean(axis=0)
std = X_train.std(axis=0) + 1e-8
X_train_norm = (X_train - mean) / std
X_val_norm = (X_val - mean) / std
X_test_norm = (X_test - mean) / std
X_train_norm = np.nan_to_num(X_train_norm, nan=0.0)
X_val_norm = np.nan_to_num(X_val_norm, nan=0.0)
X_test_norm = np.nan_to_num(X_test_norm, nan=0.0)

# 3. Preparar DataLoaders
X_train_tensor = torch.from_numpy(X_train_norm.astype(np.float32))
y_train_tensor = torch.from_numpy(y_train.astype(np.float32))
X_val_tensor = torch.from_numpy(X_val_norm.astype(np.float32))
y_val_tensor = torch.from_numpy(y_val.astype(np.float32))
X_test_tensor = torch.from_numpy(X_test_norm.astype(np.float32))
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# 4. Definir o MLP
class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
    def forward(self, x):
        return self.net(x)

class LitMLP(L.LightningModule):
    def __init__(self, input_dim):
        super().__init__()
        self.model = MLP(input_dim)
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=None)
    def forward(self, x):
        return self.model(x)
    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.unsqueeze(1)
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log('train_loss', loss)
        return loss
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.unsqueeze(1)
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log('val_loss', loss)
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

input_dim = X_train.shape[1]
model = LitMLP(input_dim)

# 5. Carregar melhor checkpoint do kaggle_02.py
checkpoint_path = "lightning_logs/checkpoints_02_04"
best_ckpt = None
if os.path.exists(checkpoint_path):
    ckpts = [f for f in os.listdir(checkpoint_path) if f.endswith(".ckpt")]
    if ckpts:
        ckpts.sort()
        best_ckpt = os.path.join(checkpoint_path, ckpts[0])
        print(f"Carregando checkpoint: {best_ckpt}")
        state = torch.load(best_ckpt)
        model.load_state_dict(state["state_dict"])

# 6. Fine-tune por mais 5 épocas
train_accs = []
val_accs = []
best_val_acc = 0.0
best_state_dict = None
for epoch in range(5):
    model.train()
    for batch in train_loader:
        x, y = batch
        y = y.unsqueeze(1)
        logits = model(x)
        loss = model.loss_fn(logits, y)
        loss.backward()
        for param in model.parameters():
            param.data -= 1e-3 * param.grad.data
        model.zero_grad()
    # Avaliar acurácia no treino
    model.eval()
    with torch.no_grad():
        all_preds = []
        all_targets = []
        for batch in train_loader:
            x, y = batch
            y = y.unsqueeze(1)
            logits = model(x)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.8).float()
            all_preds.append(preds.cpu())
            all_targets.append(y.cpu())
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        train_acc = (all_preds == all_targets).float().mean().item()
        train_accs.append(train_acc)
    # Avaliar acurácia no val
    with torch.no_grad():
        all_preds = []
        all_targets = []
        for batch in val_loader:
            x, y = batch
            y = y.unsqueeze(1)
            logits = model(x)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.8).float()
            all_preds.append(preds.cpu())
            all_targets.append(y.cpu())
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        val_acc = (all_preds == all_targets).float().mean().item()
        val_accs.append(val_acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state_dict = model.state_dict()
    print(f"Epoch {epoch+1}/5 - Train Acc: {train_acc:.4f} - Val Acc: {val_acc:.4f}")

# 7. Previsão para X_test com melhor modelo
best_model = LitMLP(input_dim)
best_model.load_state_dict(best_state_dict)
best_model.eval()
with torch.no_grad():
    logits = best_model(X_test_tensor)
    probs = torch.sigmoid(logits).cpu().numpy().flatten()
    preds = (probs > 0.8).astype(int)

# 8. Salvar previsões em CSV
pd.DataFrame({"ID": np.arange(1, len(preds)+1), "Prediction": preds}).to_csv("submission_0.8.csv", index=False)
print("Previsões salvas em submission_0.8.csv")
