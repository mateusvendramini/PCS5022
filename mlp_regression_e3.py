import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_directml
import os
import matplotlib.pyplot as plt

# Device DirectML
device = torch_directml.device()

# Carregar dados normalizados
DATASET_PATH = 'kaggle/input/pcs-5022-competicao-deca-learn-2025-task-3/dataset3_normalized.npz'
STATS_PATH = 'kaggle/input/pcs-5022-competicao-deca-learn-2025-task-3/y_train_norm_stats.npz'
data = np.load(DATASET_PATH)
stats = np.load(STATS_PATH)
mean_y = stats['mean_y'].item()
std_y = stats['std_y'].item()

X_train = torch.from_numpy(data['X_train']).float().to(device)
y_train = torch.from_numpy(data['y_train']).float().to(device)
X_val = torch.from_numpy(data['X_val']).float().to(device)
y_val = torch.from_numpy(data['y_val']).float().to(device)
X_test = torch.from_numpy(data['X_test']).float().to(device)

# Batch size e épocas
epochs = 50
batch_size = 32

# DataLoader
train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# MLP com BatchNorm
class MLPRegressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.bn = nn.BatchNorm1d(128)
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn(x)
        x = F.relu(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x.squeeze(1)

input_dim = X_train.shape[1]
model = MLPRegressor(input_dim).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
loss_fn = nn.MSELoss()

# Prealocar vetor de losses
losses = torch.zeros(epochs, device=device)

# Treinamento
for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = loss_fn(outputs, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * x_batch.size(0)
    epoch_loss /= len(train_loader.dataset)
    losses[epoch] = epoch_loss
    print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.6f}")

# Salvar modelo treinado
ckpt_dir = 'checkpoint_e3/'
os.makedirs(ckpt_dir, exist_ok=True)
torch.save(model.state_dict(), os.path.join(ckpt_dir, 'mlp_regression.pt'))

# Plotar loss por época
plt.figure(figsize=(8,5))
plt.plot(losses.cpu().numpy())
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss por época (treino)')
plt.savefig('loss_curve_e3.png')
plt.show()

# Previsão para X_test e desnormalização
def denormalize_y(y_norm):
    return y_norm * std_y + mean_y

model.eval()
with torch.no_grad():
    y_pred_norm = model(X_test)
    y_pred = denormalize_y(y_pred_norm.cpu().numpy())

# Salvar CSV de saída
ids = np.arange(1, len(y_pred)+1)
df_out = pd.DataFrame({'ID': ids, 'Prediction': y_pred})
df_out.to_csv('submission_e3_50epoch.csv', index=False)
print('Previsões salvas em submission_e3.csv')
