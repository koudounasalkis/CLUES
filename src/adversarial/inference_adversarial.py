import torch
import pandas as pd
import argparse
import numpy as np
import os
import math

from transformers import AutoModelForAudioClassification, AutoFeatureExtractor

from dataset import Dataset
from ft_utils import WeightedTrainer, define_training_args, compute_metrics
from dual_model import DualModel
from sklearn.metrics import f1_score, accuracy_score

from tqdm import tqdm
    
import warnings
warnings.filterwarnings("ignore")

""" Define Command Line Parser """
def parse_cmd_line_params():
    parser = argparse.ArgumentParser(description="batch_size")
    parser.add_argument(
        "--batch",
        help="batch size",
        default=8, 
        type=int,
        required=False)
    parser.add_argument(
        "--feature_extractor",
        help="model to use for training",
        default="facebook/wav2vec2-base",  
        type=str,                          
        required=False)   
    parser.add_argument(
        "--model",
        help="model to use for training",
        default="facebook/wav2vec2-base",  
        type=str,                          
        required=False)                     
    parser.add_argument(
        "--df_folder",
        help="path to the df folder",
        default="data/fsc",
        type=str,
        required=False) 
    parser.add_argument(
        "--output_dir",
        help="path to the output directory",
        default="results/fsc/adversarial",
        type=str,
        required=False)
    parser.add_argument(
        "--dataset",
        help="name of the dataset",
        default="fsc",
        type=str,
        required=False)
    args = parser.parse_args()
    return args



""" Read and Process Data"""
def read_data(df_folder):
    df_train = pd.read_csv(os.path.join(df_folder, 'train_data.csv'), index_col=None)
    df_test = pd.read_csv(os.path.join(df_folder, 'test_data.csv'), index_col=None)
    print("Train size: ", len(df_train))
    print("Test size: ", len(df_test))

    ## Prepare Labels
    intents = df_train['intent'].unique()
    label2id, id2label = dict(), dict()
    for i, label in enumerate(intents):
        label2id[label] = str(i)
        id2label[str(i)] = label
    num_labels = len(id2label)

    ## Create label column
    for index in range(0,len(df_train)):
        df_train.loc[index,'label'] = label2id[df_train.loc[index,'intent']]
    df_train['label'] = df_train['label'].astype(int)

    ## Test
    for index in range(0,len(df_test)):
        df_test.loc[index,'label'] = label2id[df_test.loc[index,'intent']]
    df_test['label'] = df_test['label'].astype(int)

    return df_train, df_test, num_labels, label2id, id2label
 

""" Main Program """
if __name__ == '__main__':

    ## Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ## Fix seed
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

    ## Parse command line parameters
    args = parse_cmd_line_params()
    max_duration = 4.0 

    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
        
    ## Train & Test df
    df_train, df_test, num_labels, label2id, id2label = read_data(args.df_folder)

    ## Model & Feature Extractor
    print("------------------------------------")
    print(f"Loading model from {args.model} and feature extractor from {args.feature_extractor}...")
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.feature_extractor)
    model = DualModel(
        model_name=args.feature_extractor, 
        num_classes=num_labels, 
        label2id=label2id,
        id2label=id2label,
        hidden_size=768 if 'base' in args.feature_extractor else 1024,
        local_files_only=False).to(device)
    model.load_state_dict(torch.load(args.model))
    print("Model and feature extractor loaded successfully!")
    print("------------------------------------\n")

    ## Test Dataset
    print("----------------------------------")
    print("Loading dataset...")
    test_dataset = Dataset(
        examples=df_test, 
        feature_extractor=feature_extractor, 
        max_duration=max_duration, 
        contrastive_subgroups=False,
        fine_grained_clues=False,
        augmentation=False,
        adversarial=False)
    print("Dataset loaded successfully!")
    print("----------------------------------\n")

    ## Test Dataloader
    print("----------------------------------")
    print("Creating dataloader...")
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=args.batch, 
        shuffle=False, 
        num_workers=4)
    print("Dataloader created successfully!")
    print("----------------------------------\n")

    ## Evaluate
    print("------------------------------------")
    print(f"Evaluating the model on test set...")
    print("------------------------------------\n")

    model.eval()
    preds_intents_test = []
    labels_test = []
    
    for step, batch in enumerate(tqdm(test_dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        labels_batch = batch['labels']
        del batch['labels']
        with torch.no_grad():
            output_classification, output_subgroups = model(batch)
            preds_intents_test.extend(output_classification.argmax(dim=-1).tolist())
            labels_test.extend(labels_batch.tolist())
        
    ## Compute metrics
    print("------------------------------------")
    print(f"Computing metrics on test set...")
    print("------------------------------------\n")

    ## Accuracy and f1 score
    acc = accuracy_score(labels_test, preds_intents_test)
    f1 = f1_score(labels_test, preds_intents_test, average='macro')
    print(f"Accuracy: {acc}")
    print(f"F1 score: {f1}")

    ## Save predictions
    print("------------------------------------")
    print(f"Saving predictions on test set...")
    print("------------------------------------\n")

    df_test['prediction'] = preds_intents_test == df_test['label']
    df_test['prediction'] = df_test['prediction'].astype(int)
    df_test.to_csv(os.path.join(output_dir, "df_test.csv"), index=False)