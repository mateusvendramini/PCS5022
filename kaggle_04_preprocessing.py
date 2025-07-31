import numpy as np
import torch
from transformers import AutoImageProcessor, ResNetForImageClassification
import matplotlib.pyplot as plt
import random
# Tentar usar DirectML para acelerar na GPU
try:
    import torch_directml
    device = torch_directml.device()
    print('Usando DirectML para inferência na GPU.')
except ImportError:
    device = torch.device('cpu')
    print('DirectML não disponível, usando CPU.')

# Caminho do dataset de imagens
DATASET_PATH = 'kaggle/input/dataset4/dataset_image.npz'
data = np.load(DATASET_PATH)
X_train = data['X_train']
X_test = data['X_test']

# Carregar modelo e processor
processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
model.to(device)
model.eval()


# Função para processar em batches
def get_logits_batched(images, batch_size=1000, desc=""):
    logits_list = []
    total = len(images)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = images[start:end]
        # Se necessário, converter para uint8
        if batch.dtype != np.uint8:
            batch = (batch * 255).astype(np.uint8)
        # Garantir formato [N, H, W, 3]
        if batch.shape[-1] == 1:
            batch = np.repeat(batch, 3, axis=-1)
        elif batch.shape[-1] != 3:
            raise ValueError("Imagens devem ter 1 ou 3 canais.")
        inputs = processor(list(batch), return_tensors="pt")
        for k in inputs:
            inputs[k] = inputs[k].to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        logits_list.append(outputs.logits.cpu().numpy())
        print(f"{desc} Progresso: {end}/{total} imagens processadas.")
    return np.concatenate(logits_list, axis=0)


# Classificar e salvar logits em batches
x_train_resnet50_logits = get_logits_batched(X_train, batch_size=100, desc="Treino")
x_test_resnet50_logits = get_logits_batched(X_test, batch_size=100, desc="Teste")
np.save('x_train_resnet50_logits.npy', x_train_resnet50_logits)
np.save('x_test_resnet50_logits.npy', x_test_resnet50_logits)

# Exibir 2 imagens aleatórias de cada conjunto com classificação
def show_random_images(images, logits, title):
    idxs = random.sample(range(len(images)), 2)
    for i, idx in enumerate(idxs):
        img = images[idx]
        pred = logits[idx].argmax()
        plt.subplot(1, 2, i+1)
        plt.imshow(img.squeeze() if img.shape[-1] == 1 else img)
        plt.title(f"{title} idx={idx} | Classe ResNet: {pred}")
        plt.axis('off')
    plt.show()

show_random_images(X_train, x_train_resnet50_logits, "Treino")
show_random_images(X_test, x_test_resnet50_logits, "Teste")
