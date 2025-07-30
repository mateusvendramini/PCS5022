import numpy as np
import pandas as pd
import torch
import torch_directml
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

checkpoint_path = "lightning_logs/checkpoints_03_1"
# 1. Ler o dataset
data = np.load('kaggle/input/pcs-5022-competicao-deca-learn-202-task-2/dataset2.npz')
X_train = data['X_train']
y_train = data['y_train']
X_val = data['X_val']
y_val = data['y_val']
X_test = data['X_test']
device = torch_directml.device()
#device = torch_directml.device()

# Calcular pesos das classes para lidar com desbalanceamento
classes, counts = np.unique(y_train, return_counts=True)
class_weights = counts.sum() / (len(classes) * counts)
weights_tensor = torch.tensor(class_weights, dtype=torch.float32)#.to(device)

# 2. Normalização (usando média e desvio padrão de X_train)
mean = X_train.mean(axis=0)
std = X_train.std(axis=0) + 1e-8  # evitar divisão por zero

X_train_norm = (X_train - mean) / std
X_val_norm = (X_val - mean) / std
X_test_norm = (X_test - mean) / std

# Corrige NaNs após normalização
X_train_norm = np.nan_to_num(X_train_norm, nan=0.0)
X_val_norm = np.nan_to_num(X_val_norm, nan=0.0)
X_test_norm = np.nan_to_num(X_test_norm, nan=0.0)
print("NaNs corrigidos em X_train_norm:", np.isnan(X_train_norm).sum())
print("NaNs corrigidos em X_val_norm:", np.isnan(X_val_norm).sum())
print("NaNs corrigidos em X_test_norm:", np.isnan(X_test_norm).sum())

print("X_train_norm min/max:", np.nanmin(X_train_norm), np.nanmax(X_train_norm))
print("Any NaN in X_train_norm?", np.isnan(X_train_norm).any())

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

os.makedirs(checkpoint_path, exist_ok=True)

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
        self._init_weights()
    def forward(self, x):
        return self.net(x)
    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

input_dim = X_train.shape[1]
pos_weight_value = float(weights_tensor[1])
if not np.isfinite(pos_weight_value) or pos_weight_value <= 0:
    print("WARNING: pos_weight inválido, usando 1.0")
    pos_weight_value = 1.0
#loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_value))
model = MLP(input_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Move tensores e modelo para device
X_train_tensor = X_train_tensor.to(device)
y_train_tensor = y_train_tensor.to(device)
X_val_tensor = X_val_tensor.to(device)
y_val_tensor = y_val_tensor.to(device)
X_test_tensor = X_test_tensor.to(device)
model = model.to(device)
weights_tensor = weights_tensor.to(device)

# Checkpoint loading
best_ckpt = None
if os.path.exists(checkpoint_path):
    ckpts = [f for f in os.listdir(checkpoint_path) if f.endswith(".pt")]
    if ckpts:
        ckpts.sort()
        best_ckpt = os.path.join(checkpoint_path, ckpts[0])
        print(f"Carregando checkpoint: {best_ckpt}")
        state = torch.load(best_ckpt)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        #optimizer.zero_grad()

# Fixar pesos das classes manualmente
class_0_count = 259414
class_1_count = 8231
class_weight = class_0_count / class_1_count

num_epochs = 3
train_losses = []
val_losses = []
train_accs = []
val_accs = []
best_val_loss = float('inf')
best_model_path = os.path.join(checkpoint_path, "mlp-best.pt")

print("weights_tensor:", weights_tensor)
print("y_train unique values:", np.unique(y_train))

for epoch in range(num_epochs):
    model.train()  # Garante modo treino
    running_loss = 0.0
    all_preds = []
    all_targets = []
    for batch in train_loader:
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        y = y.unsqueeze(1)
        optimizer.zero_grad()
        logits = model(x)
        # Cálculo explícito da binary cross entropy with logits
        # loss = BCEWithLogitsLoss manual
        # BCEWithLogits: -[y*log(sigmoid(x)) + (1-y)*log(1-sigmoid(x))]
        sigmoid_logits = torch.sigmoid(logits)
        eps = 1e-8
        bce = -(y * torch.log(sigmoid_logits + eps) + (1 - y) * torch.log(1 - sigmoid_logits + eps))
        # Aplica peso apenas para classe 1
        weights = torch.where(y == 1, torch.tensor(class_weight, device=device), torch.tensor(1.0, device=device))
        loss = (bce * weights).mean()
        if torch.isnan(loss):
            print("NaN loss detected! logits:", logits)
            print("y:", y)
            print("weights:", weights)
            exit(1)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x.size(0)
        probs = sigmoid_logits
        preds = (probs > 0.5).float()
        all_preds.append(preds)
        all_targets.append(y)
    epoch_loss = running_loss / len(train_loader.dataset)
    train_losses.append(epoch_loss)
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    acc = (all_preds == all_targets).float().mean().item()
    train_accs.append(acc.cpu())

    # Validation
    model.eval()
    val_running_loss = 0.0
    val_preds = []
    val_targets = []
    with torch.no_grad():
        for batch in val_loader:
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            y = y.unsqueeze(1)
            logits = model(x)
            sigmoid_logits = torch.sigmoid(logits)
            eps = 1e-8
            bce = -(y * torch.log(sigmoid_logits + eps) + (1 - y) * torch.log(1 - sigmoid_logits + eps))
            weights = torch.where(y == 1, torch.tensor(class_weight, device=device), torch.tensor(1.0, device=device))
            loss = (bce * weights).mean()
            if torch.isnan(loss):
                print("NaN val loss detected! logits:", logits)
                print("y:", y)
                print("weights:", weights)
                exit(1)
            val_running_loss += loss.item() * x.size(0)
            probs = sigmoid_logits
            preds = (probs > 0.5).float()
            val_preds.append(preds.cpu())
            val_targets.append(y.cpu())
    val_loss = val_running_loss / len(val_loader.dataset)
    val_losses.append(val_loss)
    val_preds = torch.cat(val_preds)
    val_targets = torch.cat(val_targets)
    val_acc = (val_preds == val_targets).float().mean().item()
    val_accs.append(val_acc)

    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_loss:.4f} - Val Loss: {val_loss:.4f} - Train Acc: {acc:.4f} - Val Acc: {val_acc:.4f}")

    # Save checkpoint if best val loss
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss
        }, best_model_path)

# Load best checkpoint
model.load_state_dict(torch.load(best_model_path)["model_state_dict"])
model = model.to('cpu')
model.eval()
X_test_tensor = X_test_tensor.to('cpu')

with torch.no_grad():
    logits = model(X_test_tensor)
    probs = torch.sigmoid(logits).cpu().numpy().flatten()
    preds = (probs > 0.5).astype(int)

pd.DataFrame({"ID": np.arange(1, len(preds)+1), "Prediction": preds}).to_csv("submission.csv", index=False)
print("Previsões salvas em submission.csv")

plt.figure(figsize=(8,5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss x Epoch')
plt.legend()
plt.savefig('loss_curve.png')
plt.show()

plt.figure(figsize=(8,5))
plt.plot(train_accs, label='Train Accuracy')
plt.plot(val_accs, label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy x Epoch')
plt.legend()
plt.savefig('accuracy_curve.png')
plt.show()