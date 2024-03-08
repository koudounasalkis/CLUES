import comet_ml
import torch
import pandas as pd
import argparse
import numpy as np
import os
import math
import random
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor

from dataset import Dataset
from ft_utils import WeightedTrainer, define_training_args, compute_metrics
from divergence_utils import retrieving_subgroups
    
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
        "--epochs",
        help="number of training epochs",
        default=7,      
        type=int,
        required=False)
    parser.add_argument(
        "--steps",
        help="number of steps per epoch",
        default=850,    
        type=int,
        required=False)
    parser.add_argument(
        "--gradient_accumulation_steps",
        help="number of gradient accumulation steps",
        default=4,      
        type=int,
        required=False)
    parser.add_argument(
        "--warmup_steps",
        help="number of warmup steps",
        default=5000,    
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
        default="facebook/wav2vec2-xls-r-300m",  
        type=str,                          
        required=False)                     
    parser.add_argument(
        "--df_folder",
        help="path to the df folder",
        default="data/italic",
        type=str,
        required=False) 
    parser.add_argument(
        "--output_dir",
        help="path to the output directory",
        default="results",
        type=str,
        required=False)
    parser.add_argument(
        "--dataset",
        help="name of the dataset, either 'fsc' or 'italic'",
        default="italic",
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
        default=50,
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
        default=10.0, 
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
 

""" Main Program """
if __name__ == '__main__':
    
    ## Parse command line parameters
    args = parse_cmd_line_params()
    num_epochs = args.epochs
    num_steps = args.steps
    num_wu = args.warmup_steps
    num_gas = args.gradient_accumulation_steps
    output_dir = args.output_dir
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)    

    if args.verbose:
        print("------------------------------------")
        print("Command line parameters:")
        print(f"Batch size: {args.batch}")
        print(f"Number of epochs: {num_epochs}")
        print(f"Number of steps per epoch: {num_steps}")
        print(f"Learning rate: {args.lr}")
        print(f"Model: {args.model}")
        print(f"Dataset folder: {args.df_folder}")
        print(f"Output directory: {output_dir}")
        print(f"Dataset: {args.dataset}")
        print(f"Approach: {args.approach}")
        print(f"Contrastive Learning on Intents: {args.contrastive_intent}")
        print(f"Contrastive Learning on Underperforming Subgroups: {args.contrastive_subgroups}")
        print(f"Fine-grained CLUES: {args.fine_grained_clues}")
        print(f"Adversarial Loss: {args.adversarial}")
        print(f"Data Augmentation: {args.data_augmentation}")
        print("------------------------------------\n")


    ## Set seed for reproducibility
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    ## Model & Feature Extractor
    model_checkpoint = args.model
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint)

    first_run = True

    for epoch in range(num_epochs):

        ## Train & Valid df
        if first_run:
            df_train, df_valid, num_labels, label2id, id2label = read_data(
                df_folder=args.df_folder, 
                contrastive_subgroups=False, 
                verbose=args.verbose)
        else: 
            df_train, df_valid, num_labels, label2id, id2label = read_data(
                df_folder=args.df_folder, 
                contrastive_subgroups=args.contrastive_subgroups, 
                verbose=args.verbose)

        ## Loading Model
        if not first_run:
            steps = epoch * num_steps
            model_checkpoint = os.path.join(output_dir, f"checkpoint-{steps}")
        if args.verbose:
            print("------------------------------------")
            print(f"Loading model from {model_checkpoint}")
        model = AutoModelForAudioClassification.from_pretrained(
            model_checkpoint, 
            num_labels=num_labels,
            label2id=label2id,
            id2label=id2label,
            local_files_only=not first_run)
        if args.verbose:
            print("Model loaded successfully!")
            print("------------------------------------\n")

        ## Train & Valid Datasets 
        if args.verbose:
            print("----------------------------------")
            print("Loading dataset...")
        max_duration = args.max_duration   
        if first_run:
            train_dataset = Dataset(
                examples=df_train, 
                feature_extractor=feature_extractor, 
                max_duration=max_duration, 
                contrastive_subgroups=False,
                fine_grained_clues=False,
                augmentation=args.data_augmentation,
                adversarial=False)
        else: 
            train_dataset = Dataset(
                examples=df_train, 
                feature_extractor=feature_extractor, 
                max_duration=max_duration, 
                contrastive_subgroups=args.contrastive_subgroups,
                fine_grained_clues=args.fine_grained_clues,
                augmentation=args.data_augmentation,
                adversarial=args.adversarial)
        valid_dataset = Dataset(
            examples=df_valid, 
            feature_extractor=feature_extractor, 
            max_duration=max_duration, 
            contrastive_subgroups=False,
            fine_grained_clues=False,
            augmentation=False,
            adversarial=False)
        if args.verbose:
            print("Dataset loaded successfully!")
            print("----------------------------------\n")

        ## Training Arguments
        if first_run:
            training_arguments = define_training_args(
                output_dir=output_dir, 
                batch_size=args.batch, 
                num_steps=num_steps, 
                lr=args.lr, 
                gradient_accumulation_steps=num_gas,
                warmup_steps=num_wu)
        else:
            training_arguments = define_training_args(
                output_dir=output_dir, 
                batch_size=args.batch, 
                num_steps=num_steps*(epoch+1), 
                lr=args.lr, 
                gradient_accumulation_steps=num_gas,
                warmup_steps=num_wu)

        ## Trainer 
        trainer = WeightedTrainer(
            model=model,
            args=training_arguments,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            compute_metrics=compute_metrics,
            contrastive_subgroups=args.contrastive_subgroups,
            contrastive_intent=args.contrastive_intent,
            fine_grained_clues=args.fine_grained_clues,
            adversarial=args.adversarial)

        ## Train and Evaluate
        if args.verbose:
            print("------------------------------------")
            print(f"Training the model at epoch {epoch+1}...")
            if args.contrastive_intent:
                print("Contrastive Learning on Intents")
            elif args.contrastive_subgroups:
                print("Contrastive Learning on Underperforming Subgroups")
                print(f"Approach: {args.approach}")
            elif args.fine_grained_clues:
                print("Fine-grained CLUES")
                print(f"Approach: {args.approach}")
            elif args.adversarial:
                print("Adversarial Loss")
                print(f"Approach: {args.approach}")
            elif args.data_augmentation:
                print("Data Augmentation")
            else:
                print("Classic fine-tuning")
            print("------------------------------------\n")

        if first_run:
            trainer.train()
            first_run = False
        else:
            trainer.train(resume_from_checkpoint=model_checkpoint)

        predictions = trainer.predict(valid_dataset).predictions
        df_valid['prediction'] = np.argmax(predictions, axis=1) == df_valid['label']
        df_valid['prediction'] = df_valid['prediction'].astype(int)
        df_valid.to_csv(os.path.join(output_dir, f"predictions_{epoch+1}.csv"), index=False)

        if args.contrastive_subgroups or args.adversarial:
            if args.verbose:
                print("----------------------------------")
                print("Computing subgroups...")
            retrieving_subgroups(
                args.df_folder, 
                df_valid, 
                output_dir,
                args.dataset, 
                args.approach, 
                min_sup=args.min_support,
                num_problematic_subgroups=args.num_problematic_subgroups,
                fine_grained_clues=args.fine_grained_clues,
                feature_extractor=feature_extractor,
                max_duration=max_duration,
                model_checkpoint=model_checkpoint,
                verbose=args.verbose
                )
            if args.verbose:   
                print("----------------------------------\n")

    if args.verbose:
        print("Training completed successfully!")
        print("------------------------------------\n")