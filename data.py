import numpy as np
import pandas as pd

# Carregar o arquivo .npz
data = np.load('kaggle/input/pcs-5022-competicao-deca-learn-202-task-2/dataset2.npz')

# Listar as chaves disponíveis no arquivo
print("Chaves disponíveis no arquivo .npz:", data.files)

# Exemplo: converter X_val para DataFrame (se for possível)
if 'X_test' in data:
    X_test = data['X_test']
    # Se X_val for 2D, pode ser convertido diretamente
    if X_test.ndim == 2:
        df_X = pd.DataFrame(X_test)
        print("Colunas de X_val:", df_X.columns)
        print(df_X.head())
    else:
        print("X_val não é 2D, shape:", X_test.shape)
else:
    print("X_val não encontrado no arquivo.")

# Exemplo: converter y_val para DataFrame
if 'X_test' in X_test:
    X_test = data['X_test']
    df_y = pd.DataFrame(X_test, columns=['X_test'])
    print(df_y.head())
else:
    print("y_val não encontrado no arquivo.")

# Estatísticas básicas
if 'X_test' in data and X_test.ndim == 2:
    print("Descrição estatística de X_val:")
    print(df_X.describe())

# Avaliar a distribuição das classes em 'y_train' (classe binária)
if 'y_train' in data:
    y_train = data['y_train']
    # Se for 1D, pode ser convertido diretamente
    if y_train.ndim == 1:
        df_y_train = pd.DataFrame(y_train, columns=['y_train'])
        print("Distribuição das classes em y_train:")
        print(df_y_train['y_train'].value_counts())
    else:
        print("y_train não é 1D, shape:", y_train.shape)
else:
    print("y_train não encontrado no arquivo.")