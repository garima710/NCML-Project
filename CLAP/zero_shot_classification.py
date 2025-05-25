
from msclap import CLAP
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score

import sys
import os

sys.path.append(r"C:/Users/garim/OneDrive/Documents/GitHub/NCML-Project")

# Now import your module
from esc50_dataset import ESC50


# Load dataset
root_path = "root_path" # Folder with ESC-50-master/
dataset = ESC50(root=root_path, download=True) #If download=False code assumes base_folder='ESC-50-master' in esc50_dataset.py
prompt = 'this is the sound of '
y = [prompt + x for x in dataset.classes]

# Load and initialize CLAP
clap_model = CLAP(version = '2023', use_cuda=False)

# Computing text embeddings
text_embeddings = clap_model.get_text_embeddings(y)

# Computing audio embeddings
y_preds, y_labels = [], []
for i in tqdm(range(len(dataset))):
    x, _, one_hot_target = dataset.__getitem__(i)
    audio_embeddings = clap_model.get_audio_embeddings([x], resample=True)
    similarity = clap_model.compute_similarity(audio_embeddings, text_embeddings)
    y_pred = F.softmax(similarity.detach().cpu(), dim=1).numpy()
    y_preds.append(y_pred)
    y_labels.append(one_hot_target.detach().cpu().numpy())


y_labels, y_preds = np.concatenate(y_labels, axis=0), np.concatenate(y_preds, axis=0)
acc = accuracy_score(np.argmax(y_labels, axis=1), np.argmax(y_preds, axis=1))
print('ESC50 Accuracy {}'.format(acc))


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# conf_mat = confusion_matrix(np.argmax(y_labels, axis=1), np.argmax(y_preds, axis=1))
# plt.figure(figsize=(12, 10))
# sns.heatmap(conf_mat, annot=True, fmt='d', xticklabels=dataset.classes, yticklabels=dataset.classes, cmap="Blues")
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.title('Confusion Matrix')
# plt.xticks(rotation=90)
# plt.yticks(rotation=0)
# plt.tight_layout()
# plt.show()


# # Save the confusion matrix as an image file
# plt.savefig('confusion_matrix.png')

from sklearn.manifold import TSNE

# # Project audio embeddings to 2D
# audio_embeddings_all = [clap_model.get_audio_embeddings([dataset[i][0]], resample=True).detach().cpu().numpy()[0] for i in range(len(dataset))]
# labels = [np.argmax(dataset[i][2].detach().cpu().numpy()) for i in range(len(dataset))]

# tsne = TSNE(n_components=2, perplexity=30, random_state=42)
# embeddings_2d = tsne.fit_transform(audio_embeddings_all)

# plt.figure(figsize=(10, 8))
# scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='tab20', alpha=0.7)
# plt.title('t-SNE of Audio Embeddings')
# plt.colorbar(scatter, ticks=range(len(dataset.classes)), label='Class')
# plt.clim(-0.5, len(dataset.classes)-0.5)
# plt.show()

top1 = np.argmax(y_preds, axis=1) == np.argmax(y_labels, axis=1)
top5 = [np.argmax(y_labels[i]) in y_preds[i].argsort()[-5:] for i in range(len(y_preds))]

plt.bar(['Top-1 Accuracy', 'Top-5 Accuracy'], [np.mean(top1), np.mean(top5)])
plt.ylim(0, 1)
plt.title('Top-N Accuracy')
plt.ylabel('Accuracy')
plt.show()

class_accuracies = []
for i in range(len(dataset.classes)):
    indices = [j for j, label in enumerate(np.argmax(y_labels, axis=1)) if label == i]
    correct = sum(np.argmax(y_preds[j]) == i for j in indices)
    class_accuracies.append(correct / len(indices))

plt.figure(figsize=(12, 6))
plt.bar(dataset.classes, class_accuracies)
plt.xticks(rotation=90)
plt.title('Accuracy per Class')
plt.ylabel('Accuracy')
plt.tight_layout()
plt.show()
