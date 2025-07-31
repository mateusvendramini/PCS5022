import numpy as np
import matplotlib.pyplot as plt

# Caminho do dataset de imagens
DATASET_PATH = 'kaggle/input/dataset4/dataset_image.npz'

data = np.load(DATASET_PATH)

# Listar as chaves disponíveis
print('Chaves disponíveis:', data.files)

# Carregar os dados
X_train = data['X_train']
y_train = data['y_train']
X_test = data['X_test']

# Verificar proporções das classes em y_train
classes, counts = np.unique(y_train, return_counts=True)
proportions = counts / counts.sum()
print('Proporções das classes em y_train:')
for c, p, cnt in zip(classes, proportions, counts):
    print(f'Classe {int(c)}: {cnt} amostras ({p:.2%})')

# Informações das imagens
num_train = X_train.shape[0]
num_test = X_test.shape[0]
img_shape = X_train.shape[1:]

print(f'Número de imagens de treino: {num_train}')
print(f'Número de imagens de teste: {num_test}')
print(f'Resolução das imagens: {img_shape}')
print(f'Número de canais (cores): {img_shape[-1] if len(img_shape) == 3 else 1}')

# Exibir 16 imagens de teste
plt.figure(figsize=(16, 8))
for i in range(16):
    plt.subplot(2, 8, i+1)
    img = X_test[i]
    # Se y_train e X_test estão alinhados, pode-se usar y_train[i], mas normalmente X_test não tem label. Aqui, só para exemplo:
    label = None
    if i < len(y_train):
        label = int(y_train[i])
    title = f'Teste {i+1}' + (f' | Classe: {label}' if label is not None else '')
    if img_shape[-1] == 1:
        plt.imshow(img.squeeze(), cmap='gray')
    else:
        plt.imshow(img)
    plt.title(title)
    plt.axis('off')
plt.tight_layout()
plt.show()
