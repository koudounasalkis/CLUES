import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils import class_weight
from pytorch_metric_learning import losses
import pandas as pd
import argparse
from transformers import TrainingArguments, Trainer
import os 
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from sklearn.utils import shuffle, class_weight
from transformers import AutoTokenizer, Trainer, TrainerCallback, TrainerState
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import mean_squared_error
from tqdm import tqdm


""" Define training arguments """ 
def define_training_args(
    output_dir, 
    batch_size, 
    num_steps=500, 
    lr=1.0e-4, 
    gradient_accumulation_steps=1, 
    warmup_steps=0
    ): 
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        evaluation_strategy="steps",
        save_strategy="steps",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        per_device_eval_batch_size=batch_size,
        gradient_checkpointing=True,
        max_steps=num_steps,
        warmup_steps=warmup_steps,
        logging_steps=100,
        eval_steps=num_steps,
        save_steps=num_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        fp16=True,
        fp16_full_eval=True,
        dataloader_num_workers=16,
        dataloader_pin_memory=True,
        remove_unused_columns=False)
    return training_args


""" Define Metric """
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average='macro')
    print('Accuracy: ', acc, 'F1 Macro: ', f1_macro)
    return { 'accuracy': acc, 'f1_macro': f1_macro }



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
        "--epochs",
        help="number of training epochs",
        default=5,      
        type=int,
        required=False)
    parser.add_argument(
        "--steps",
        help="number of steps per epoch",
        default=500,   
        type=int,
        required=False)
    parser.add_argument(
        "--gradient_accumulation_steps",
        help="number of gradient accumulation steps",
        default=1,     
        type=int,
        required=False)
    parser.add_argument(
        "--warmup_steps",
        help="number of warmup steps",
        default=500,    
        type=int,
        required=False)
    parser.add_argument(
        "--lr",
        help="learning rate",
        default=1e-4,
        type=float,
        required=False)
    parser.add_argument(
        "--model",
        help="model to use for training, \
            either 'facebook/wav2vec2-base' for FSC \
            or 'facebook/wav2vec2-xls-r-300m' for ITALIC",
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
        help="name of the dataset, either 'fsc' or 'italic'",
        default="fsc",
        type=str,
        required=False)
    parser.add_argument(
        "--approach",
        help="approach to be used for subgroups identification: \
            one of 'clustering', 'divexplorer'",
        default="divexplorer",
        type=str,
        required=False)
    parser.add_argument(
        "--min_support",
        help="minimum support",
        default=0.03,
        type=float,
        required=False)
    parser.add_argument(
        "--contrastive_intent",
        help="whether to use contrastive learning",
        action="store_true",
        required=False)
    parser.add_argument(
        "--contrastive_subgroups",
        help="whether to use contrastive learning with underperforming subgroups",
        action="store_true",
        required=False)
    parser.add_argument(
        "--fine_grained_clues",
        help="whether to use CLUES, fine-grained contrastive learning with underperforming subgroups",
        action="store_true",
        required=False)
    parser.add_argument(
        "--adversarial",
        help="whether to use additional adversarial loss in training",
        action="store_true",
        required=False)
    parser.add_argument(
        "--data_augmentation",
        help="whether to use data augmentation",
        action="store_true",
        required=False)
    parser.add_argument(
        "--verbose",
        help="whether to print logging information",
        action="store_true",
        required=False)
    parser.add_argument(
        "--num_problematic_subgroups",
        help="Number of problematic subgroups to be used for mitigation",
        default=10,
        type=int,
        required=False)
    parser.add_argument(
        "--seed",
        help="Seed to be used for reproducibility",
        default=42,
        type=int,
        required=False)
    parser.add_argument(
        "--max_duration",
        help="Maximum duration of audio files",
        default=4.0, 
        type=float,
        required=False)
    args = parser.parse_args()
    return args



""" Read and Process Data"""
def read_data(df_folder, contrastive_subgroups=False, verbose=False):
    if contrastive_subgroups:
        df_train = pd.read_csv(
            os.path.join(df_folder, 'new_data', 'train_data.csv'), 
            index_col=None)
    else:
        df_train = pd.read_csv(
            os.path.join(df_folder, 'train_data.csv'), 
            index_col=None)
    df_valid = pd.read_csv(
        os.path.join(df_folder, 'valid_data.csv'), 
        index_col=None)
    if verbose:
        print("Train size: ", len(df_train))
        print("Valid size: ", len(df_valid))

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
    df_train.to_csv(os.path.join(df_folder, 'train_data.csv'), index=False)

    ## Test
    for index in range(0,len(df_valid)):
        df_valid.loc[index,'label'] = label2id[df_valid.loc[index,'intent']]
    df_valid['label'] = df_valid['label'].astype(int)
    df_valid.to_csv(os.path.join(df_folder, 'valid_data.csv'), index=False)

    return df_train, df_valid, num_labels, label2id, id2label