import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

# Carregar os dados
train_logits = np.load('kaggle/input/dataset4/x_train_resnet50_logits.npy')
test_logits = np.load('kaggle/input/dataset4/x_test_resnet50_logits.npy')
data = np.load('kaggle/input/dataset4/dataset_image.npz')
y_train = data['y_train']

# Separar treino/validação (mesmo split)
X_train, X_val, y_train_split, y_val = train_test_split(
    train_logits, y_train, test_size=0.2, random_state=42, stratify=y_train
)

# Treinar XGBoost
model = XGBClassifier(
    n_estimators=500,
    max_depth=10,
    learning_rate=0.01,
    use_label_encoder=False,
    eval_metric='logloss',
    early_stopping_rounds=10,
    n_threads=12
)
model.fit(X_train, y_train_split, eval_set=[(X_val, y_val)], verbose=True)

# Avaliar desempenho
train_preds = model.predict(X_train)
val_preds = model.predict(X_val)
train_acc = (train_preds == y_train_split).mean()
val_acc = (val_preds == y_val).mean()
print(f"Acurácia Treino: {train_acc:.4f} - Acurácia Validação: {val_acc:.4f}")

# Plotar curva de acurácia e logloss
results = model.evals_result()
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(results['validation_0']['logloss'], label='Val Logloss')
plt.xlabel('Época')
plt.ylabel('Logloss')
plt.legend()
plt.subplot(1,2,2)
plt.plot(results['validation_0']['logloss'], label='Val Logloss')
plt.xlabel('Época')
plt.ylabel('Logloss')
plt.legend()
plt.tight_layout()
plt.show()

# Previsões para o conjunto de teste
test_preds = model.predict(test_logits)

# Gerar CSV
ids = np.arange(1, len(test_preds)+1)
df = pd.DataFrame({'ID': ids, 'Prediction': test_preds.astype(int)})
df.to_csv('submission_xgboost_resnet50.csv', index=False)
print('Arquivo submission_xgboost_resnet50.csv gerado.')