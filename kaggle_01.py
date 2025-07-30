import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torch.utils.data import DataLoader, TensorDataset
from lightning.pytorch.callbacks import ModelCheckpoint
import os
import matplotlib.pyplot as plt

# 1. Ler o dataset
data = np.load('kaggle/input/pcs-5022-competicao-deca-learn-202-task-1/dataset1.npz')
X_train = data['X_train']
y_train = data['y_train']
X_val = data['X_val']
y_val = data['y_val']
X_test = data['X_test']

# 2. Normalização (usando média e desvio padrão de X_train)
mean = X_train.mean(axis=0)
std = X_train.std(axis=0) + 1e-8  # evitar divisão por zero

X_train_norm = (X_train - mean) / std
X_val_norm = (X_val - mean) / std
X_test_norm = (X_test - mean) / std

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
    def forward(self, x):
        return self.model(x)
    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.unsqueeze(1)
        logits = self(x)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        self.log('train_loss', loss)
        return loss
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.unsqueeze(1)
        logits = self(x)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        self.log('val_loss', loss)
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

input_dim = X_train.shape[1]
model = LitMLP(input_dim)

# 5. Checkpoint callback
checkpoint_callback = ModelCheckpoint(
    dirpath="lightning_logs/checkpoints3",
    filename="mlp-{epoch:02d}-{val_loss:.2f}",
    save_top_k=1,
    monitor="val_loss",
    mode="min"
)
# 5.1 Carregar checkpoint se existir
best_ckpt = None
if os.path.exists("lightning_logs/checkpoints3"):
    ckpts = [f for f in os.listdir("lightning_logs/checkpoints3") if f.endswith(".ckpt")]
    if ckpts:
        # Pega o melhor checkpoint salvo
        ckpts.sort()
        best_ckpt = os.path.join("lightning_logs/checkpoints3", ckpts[0])
        print(f"Carregando checkpoint: {best_ckpt}")
        state = torch.load(best_ckpt)
        model.load_state_dict(state["state_dict"])
# 6. Treinar o modelo

# 6. Treinar o modelo e salvar losses
train_losses = []
val_losses = []
train_accs = []
val_accs = []
class LossLogger(L.Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        train_loss = trainer.callback_metrics.get("train_loss")
        if train_loss is not None:
            train_losses.append(train_loss.cpu().item())
        # Calcular acurácia no conjunto de treino
        all_preds = []
        all_targets = []
        for batch in train_loader:
            x, y = batch
            y = y.unsqueeze(1)
            logits = pl_module(x)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            all_preds.append(preds.cpu())
            all_targets.append(y.cpu())
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        acc = (all_preds == all_targets).float().mean().item()
        train_accs.append(acc)
    def on_validation_epoch_end(self, trainer, pl_module):
        val_loss = trainer.callback_metrics.get("val_loss")
        if val_loss is not None:
            val_losses.append(val_loss.cpu().item())
        # Calcular acurácia no conjunto de validação
        all_preds = []
        all_targets = []
        for batch in val_loader:
            x, y = batch
            y = y.unsqueeze(1)
            logits = pl_module(x)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            all_preds.append(preds.cpu())
            all_targets.append(y.cpu())
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        acc = (all_preds == all_targets).float().mean().item()
        val_accs.append(acc)

loss_logger = LossLogger()
trainer = L.Trainer(
    max_epochs=10,
    callbacks=[checkpoint_callback, loss_logger]
)
trainer.fit(model, train_loader, val_loader)

# 7. Carregar melhor checkpoint
best_model_path = checkpoint_callback.best_model_path
trained_model = LitMLP(input_dim)
trained_model.load_state_dict(torch.load(best_model_path)["state_dict"])
trained_model.eval()

# 8. Previsão para X_test
with torch.no_grad():
    logits = trained_model(X_test_tensor)
    probs = torch.sigmoid(logits).cpu().numpy().flatten()
    preds = (probs > 0.5).astype(int)

# 9. Salvar previsões em CSV com colunas ID e Prediction
pd.DataFrame({"ID": np.arange(1, len(preds)+1), "Prediction": preds}).to_csv("submission.csv", index=False)
print("Previsões salvas em submission.csv")

# 10. Plotar loss x epoch
plt.figure(figsize=(8,5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss x Epoch')
plt.legend()
plt.savefig('loss_curve.png')
plt.show()

# 11. Plotar acurácia x época
plt.figure(figsize=(8,5))
plt.plot(train_accs, label='Train Accuracy')
plt.plot(val_accs, label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy x Epoch')
plt.legend()
plt.savefig('accuracy_curve.png')
plt.show()