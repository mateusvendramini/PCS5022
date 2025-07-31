import numpy as np
import pandas as pd
import os

# Caminho do dataset do terceiro exercício
DATASET_PATH = 'kaggle/input/pcs-5022-competicao-deca-learn-2025-task-3/dataset3.npz'
SAVE_PATH = 'kaggle/input/pcs-5022-competicao-deca-learn-2025-task-3/'

data = np.load(DATASET_PATH)

# Listar as chaves disponíveis
print('Chaves disponíveis:', data.files)

# Carregar os dados

def check_and_fix_nan(arr, name):
    has_nan = np.isnan(arr).any()
    print(f"{name} contém NaN?", has_nan)
    arr_fixed = np.nan_to_num(arr, nan=0.0)
    return arr_fixed

X_train = check_and_fix_nan(data['X_train'], 'X_train')
y_train = check_and_fix_nan(data['y_train'], 'y_train')
X_val = check_and_fix_nan(data['X_val'], 'X_val')
y_val = check_and_fix_nan(data['y_val'], 'y_val')
X_test = check_and_fix_nan(data['X_test'], 'X_test')

# Explorar X_train e y_train
print('X_train shape:', X_train.shape)
print('y_train shape:', y_train.shape)
print('X_train stats:')
print(pd.DataFrame(X_train).describe())
print('y_train stats:')
print(pd.DataFrame(y_train).describe())

# Normalização usando média e desvio padrão de X_train e y_train
mean_X = X_train.mean(axis=0)
std_X = X_train.std(axis=0) + 1e-8
mean_y = y_train.mean()
std_y = y_train.std() + 1e-8

# Normalizar X e y
X_train_norm = (X_train - mean_X) / std_X
X_val_norm = (X_val - mean_X) / std_X
X_test_norm = (X_test - mean_X) / std_X
y_train_norm = (y_train - mean_y) / std_y
y_val_norm = (y_val - mean_y) / std_y

# Salvar vetores normalizados
np.savez(os.path.join(SAVE_PATH, 'dataset3_normalized.npz'),
         X_train=X_train_norm,
         y_train=y_train_norm,
         X_val=X_val_norm,
         y_val=y_val_norm,
         X_test=X_test_norm)

# Salvar média e desvio padrão de y_train
np.savez(os.path.join(SAVE_PATH, 'y_train_norm_stats.npz'), mean_y=mean_y, std_y=std_y)

print('Vetores normalizados e estatísticas salvos em', SAVE_PATH)
