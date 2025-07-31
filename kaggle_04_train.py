import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

# Carregar os dados
train_logits = np.load('kaggle/input/dataset4/x_train_resnet50_logits.npy')
test_logits = np.load('kaggle/input/dataset4/x_test_resnet50_logits.npy')
data = np.load('kaggle/input/dataset4/dataset_image.npz')
y_train = data['y_train']

# Separar treino/validação
X_train, X_val, y_train_split, y_val = train_test_split(
    train_logits, y_train, test_size=0.2, random_state=42, stratify=y_train)

# Converter para tensores
X_train = torch.tensor(X_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_train_split = torch.tensor(y_train_split, dtype=torch.float32).unsqueeze(1)
y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
X_test = torch.tensor(test_logits, dtype=torch.float32)

# MLP
class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x)

model = MLP(input_dim=X_train.shape[1])
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Treinamento
num_epochs = 400
train_losses, val_losses = [], []
train_accs, val_accs = [], []

for epoch in range(num_epochs):
    # Treino
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train_split)
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())
    preds = (outputs.detach().numpy() > 0.5).astype(int)
    acc = (preds == y_train_split.numpy()).mean()
    train_accs.append(acc)

    # Validação
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        val_loss = criterion(val_outputs, y_val)
        val_losses.append(val_loss.item())
        val_preds = (val_outputs.numpy() > 0.5).astype(int)
        val_acc = (val_preds == y_val.numpy()).mean()
        val_accs.append(val_acc)
    print(f"Época {epoch+1}/{num_epochs} - Loss Treino: {loss.item():.4f} - Acc Treino: {acc:.4f} - Loss Val: {val_loss.item():.4f} - Acc Val: {val_acc:.4f}")

# Plotar loss e acurácia
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Época')
plt.ylabel('Loss')
plt.legend()
plt.subplot(1,2,2)
plt.plot(train_accs, label='Train Acc')
plt.plot(val_accs, label='Val Acc')
plt.xlabel('Época')
plt.ylabel('Acurácia')
plt.legend()
plt.tight_layout()
plt.show()

# Previsões para o conjunto de teste
model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    test_preds = (test_outputs.numpy() > 0.5).astype(int).flatten()

# Gerar CSV
ids = np.arange(1, len(test_preds)+1)
df = pd.DataFrame({'ID': ids, 'Prediction': test_preds})
df.to_csv('submission_mlp_resnet50.csv', index=False)
print('Arquivo submission_mlp_resnet50.csv gerado.')
