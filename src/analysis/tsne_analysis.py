
import transformers
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
import torch
import torchaudio
import pandas as pd
import argparse
import os
from tqdm import tqdm

from dataset import Dataset

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='TSNE on the SLU dataset.')
parser.add_argument(
    '--df_folder', 
    type=str, 
    default='results/fsc/', 
    help='Path to the df folder.')
parser.add_argument(
    '--feature_extractor', 
    type=str, 
    default="facebook/wav2vec2-base", 
    help='Feature extractor.')
parser.add_argument(
    '--model_name_or_path', 
    type=str, 
    default="facebook/wav2vec2-base", 
    help='Model name or path.')
parser.add_argument(
    '--max_duration', 
    type=float, 
    default=4.0, 
    help='Max duration of the audio files.')
parser.add_argument(
    '--device', 
    type=str, 
    default="cuda", 
    help='Device.')
parser.add_argument(
    '--save_folder', 
    type=str, 
    default="analysis/", 
    help='Folder to save the results.')
parser.add_argument(
    '--show_labels', 
    action='store_true', 
    help='Whether the image shows the labels or not.')
parser.add_argument(
    '--approach', 
    type=str, 
    default='both', 
    help='Whether the labels are intents, problematic_subgroups or both.')
parser.add_argument(
    '--dataset', 
    type=str, 
    default='fsc', 
    help='fsc or italic')
args = parser.parse_args()

## Define the top K intents
K = 5 

## Create save folder if it does not exist
if not os.path.exists(args.save_folder):
    os.makedirs(args.save_folder)

# ------------------------ #
# Parse the data.
# ------------------------ #
feature_extractor = AutoFeatureExtractor.from_pretrained(args.feature_extractor)
df_test = pd.read_csv(os.path.join(args.df_folder, 'df_test.csv'), index_col=None)
test_ds = Dataset(
    df_test, 
    feature_extractor, 
    args.max_duration,
    contrastive_subgroups=False
    )

test_dl = torch.utils.data.DataLoader(
    test_ds,
    batch_size=16,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    )

## Load the model
model = AutoModelForAudioClassification.from_pretrained(
    args.model_name_or_path,
    output_hidden_states=True,
    ).to(args.device)


import pickle
## Load the embeddings if they exist
if os.path.exists(f"{args.save_folder}/embeddings.pkl"):
    with open(f"{args.save_folder}/embeddings.pkl", "rb") as f:
        embeddings = pickle.load(f)
    with open(f"{args.save_folder}/labels.pkl", "rb") as f:
        labels = pickle.load(f)
    with open(f"{args.save_folder}/predicted_labels.pkl", "rb") as f:
        predicted_labels = pickle.load(f)
    with open(f"{args.save_folder}/predictions.pkl", "rb") as f:
        predictions = pickle.load(f)
else:
    ## Extract embeddings
    embeddings, labels = [], []
    for batch in tqdm(test_dl, desc="Extracting embeddings", ncols=100):
        audio, label = batch["input_values"].to(args.device), batch["labels"].to(args.device)
        with torch.no_grad():
            output = model(audio)
            embedding = output.hidden_states[-1]
            embedding = embedding.mean(dim=1).squeeze(1).detach().cpu().numpy()
            embeddings.extend(embedding)
            labels.extend(label.cpu().numpy())
    predicted_labels = df_test['predicted_label'].tolist()
    predictions = df_test['prediction'].tolist()

    ## Save pkls
    with open(f"{args.save_folder}/embeddings.pkl", "wb") as f:
        pickle.dump(embeddings, f)
    with open(f"{args.save_folder}/labels.pkl", "wb") as f:
        pickle.dump(labels, f)
    with open(f"{args.save_folder}/predicted_labels.pkl", "wb") as f:
        pickle.dump(predicted_labels, f)
    with open(f"{args.save_folder}/predictions.pkl", "wb") as f:
        pickle.dump(predictions, f)

## Extract clustering image with t-SNE
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

embeddings = np.array(embeddings)
labels = np.array(labels)
predicted_labels = np.array(predicted_labels)
predictions = np.array(predictions)

import os 
if not os.path.exists("analysis"):
    os.makedirs("analysis")

# --------------------------- t-SNE --------------------------- 
import seaborn as sns
sns.set_theme()

print("Computing t-SNE")
# if not pickled, compute t-SNE
if not os.path.exists(f"{args.save_folder}/tsne.pkl"):
    tsne = TSNE(n_components=2, random_state=0)
    tsne_embeddings = tsne.fit_transform(embeddings)
    with open(f"{args.save_folder}/tsne.pkl", "wb") as f:
        pickle.dump(tsne_embeddings, f)
else:
    with open(f"{args.save_folder}/tsne.pkl", "rb") as f:
        tsne_embeddings = pickle.load(f)

if args.approach == 'intent':
    ## Intents: the labels are the intents
    df_train = pd.read_csv(f'data/{args.dataset}/train_data.csv', index_col=None)
    intents = df_train['intent'].unique()
    label_to_id, id_to_label = dict(), dict()
    for i, label in enumerate(intents):
        label_to_id[label] = str(i)
        id_to_label[str(i)] = label
    num_labels = len(id_to_label)
    label_names = intents
    label_names = sorted(label_names) 
elif args.approach == 'problematic_subgroups':
    ## The labels are 1 (correct) and 0 (incorrect), derived from the predictions
    label_names = ["correct", "incorrect"]
elif args.approach == 'both':
    ### Filter out the correct predictions and get the top K intents with the most occurrences
    df_test = df_test[df_test['prediction'] == 0]
    intents = df_test['intent'].unique()
    intents = sorted(intents, key=lambda x: df_test['intent'].value_counts()[x], reverse=True)[:K]
    ### Create label column
    label_to_id, id_to_label = dict(), dict()
    for i, label in enumerate(intents):
        label_to_id[label] = str(i)
        id_to_label[str(i)] = label
    num_labels = len(id_to_label)
    label_names = ["correct", "incorrect"]
    label_names = [f"{label_name} ({intent})" for label_name in label_names for intent in intents]
else:
    raise ValueError("Approach not recognized.")


## Create a color map
plt.figure(figsize=(20, 15))
plt.rc('font', size=30)
cmap = plt.get_cmap('tab20', len(label_names))
colors = ['blue', 'red', 'green', 'orange', 'purple']

## Scatter plot with custom colors based on labels
if args.approach == 'intent':
    for i, label_name in enumerate(label_names):
        label_indices = np.where(labels == i)
        plt.scatter(
            tsne_embeddings[label_indices, 0], 
            tsne_embeddings[label_indices, 1], 
            s=8, 
            c=[cmap(i)], 
            label=label_name
            )
elif args.approach == 'problematic_subgroups':
    for i, label_name in enumerate(label_names):
        label_indices = np.where(predictions == i)
        plt.scatter(
            tsne_embeddings[label_indices, 0], 
            tsne_embeddings[label_indices, 1], 
            s=16, 
            c=[cmap(i)], 
            label=label_name
            )
elif args.approach == 'both':
    for i, label_name in enumerate(label_names):
        label_indices = np.where(labels == i)
        ## If the first part of the label is the same, use the same color and change the marker
        if i % 2 == 0:
            plt.scatter(
                tsne_embeddings[label_indices, 0], 
                tsne_embeddings[label_indices, 1], 
                s=20, 
                c=colors[i//2],
                label=label_name,
                )
        else:
            plt.scatter(
                tsne_embeddings[label_indices, 0], 
                tsne_embeddings[label_indices, 1], 
                s=50, 
                c=colors[i//2 - 1],
                label=label_name,
                marker='x'
                )

## Create new names for the labels
new_names = []
for i, label_name in enumerate(label_names):
    split = label_name.split(" (")
    new_name = split[1][:-1] + " (" + split[0] + ")"
    new_names.append(new_name)
new_names = sorted(new_names)

## Annotate with labels
if args.show_labels:
    mean_coordinates = []
    first_coordinates = []
    ### Get the mean coordinates of each label
    for i in range(len(new_names)):
        label_indices = np.where(labels == i)
        label_embeddings = tsne_embeddings[label_indices]
        mean_x = np.mean(label_embeddings[:, 0])
        mean_y = np.mean(label_embeddings[:, 1])
        mean_coordinates.append((mean_x, mean_y))
        first_coordinates.append((label_embeddings[0, 0], label_embeddings[0, 1]))
    ### Annotate with mean coordinates
    for i, (mean_x, mean_y) in enumerate(mean_coordinates):
        plt.annotate(new_names[i], (mean_x, mean_y), fontsize=20)

## Create the legend with custom colors
legend_handles = []
for i, new_name in enumerate(new_names):
    legend_handles.append(plt.Line2D(
        [0], [0], 
        marker='o' if i % 2 == 0 else 'X',
        color='w', 
        label=new_name, 
        markersize=20, 
        markerfacecolor=colors[i//2],
        ))

plt.legend(
    handles=legend_handles, 
    loc='upper center', 
    bbox_to_anchor=(0.5, -1.05),
    fancybox=True, 
    shadow=True, 
    ncol=5,
    facecolor='white',
    )
plt.xticks(np.arange(-60, 90, 20), fontsize=30)
plt.yticks(np.arange(-80, 90, 20), fontsize=30)
plt.tight_layout()
plt.savefig(f"{args.save_folder}/tsne-legend.png")
plt.savefig(f"{args.save_folder}/tsne-legend.pdf")
plt.clf()

## Compute the distance matrix between the embeddings
from sklearn.metrics.pairwise import cosine_distances

distances = cosine_distances(embeddings)
mean_distances = []
for i in range(len(label_names)):
    label_indices = np.where(labels == i)
    label_distances = distances[label_indices]
    label_distances = label_distances[:, label_indices]
    label_distances = label_distances[label_distances != 0]
    mean_distances.append(np.mean(label_distances))
print("---------------")
print(args.model_name_or_path)
print("Embeddings")
print("Mean distance: ", np.mean(mean_distances))
print("Std distance: ", np.std(mean_distances))

## Compute the cosine similarity between the embeddings
from sklearn.metrics.pairwise import cosine_similarity

similarities = cosine_similarity(embeddings)
mean_similarities = []
for i in range(len(label_names)):
    label_indices = np.where(labels == i)
    label_similarities = similarities[label_indices]
    label_similarities = label_similarities[:, label_indices]
    label_similarities = label_similarities[label_similarities != 1]
    mean_similarities.append(np.mean(label_similarities))
print("Embeddings")
print("Mean similarity: ", np.mean(mean_similarities))
print("Std similarity: ", np.std(mean_similarities))
print("---------------\n")

## Compute silhouette score
from sklearn.metrics import silhouette_score
silhouette = silhouette_score(embeddings, labels)
print("Silhouette score: ", silhouette)

## Compute Silhouette score considering only the problematic subgroups
if args.approach == 'both':
    silhouette_problematic = silhouette_score(embeddings[predictions == 0], labels[predictions == 0])
    print("Silhouette score (problematic): ", silhouette_problematic)
    