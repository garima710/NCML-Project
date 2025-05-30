# CLAP Zero-Shot Predictions
from msclap import CLAP
import torch.nn.functional as F

# Define classes for zero-shot
# Should be in lower case and can be more than one word
classes = ['coughing','sneezing','drinking sipping', 'breathing', 'brushing teeth']
ground_truth = ['coughing']
# Add prompt
prompt = 'this is a sound of '
class_prompts = [prompt + x for x in classes]
#Load audio files
audio_files = ['audio_file']

# Load and initialize CLAP
# Setting use_cuda = True will load the model on a GPU using CUDA
clap_model = CLAP(version = '2023', use_cuda=False)

# compute text embeddings from natural text
text_embeddings = clap_model.get_text_embeddings(class_prompts)

# compute the audio embeddings from an audio file
audio_embeddings = clap_model.get_audio_embeddings(audio_files, resample=True)

# compute the similarity between audio_embeddings and text_embeddings
similarity = clap_model.compute_similarity(audio_embeddings, text_embeddings)

similarity = F.softmax(similarity, dim=1)
values, indices = similarity[0].topk(5)

# Print the results
print("Ground Truth: {}".format(ground_truth))
print("Top predictions:\n")
for value, index in zip(values, indices):
    print(f"{classes[index]:>16s}: {100 * value.item():.2f}%")

