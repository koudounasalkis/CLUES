import comet_ml
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import math
import random
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor

from dataset import Dataset
from ft_utils_dual import parse_cmd_line_params, read_data
from divergence_utils_dual import retrieving_subgroups
from dual_model import DualModel
    
import warnings
warnings.filterwarnings("ignore")


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

    ## Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        print(f"Number of problematic subgroups: {args.num_problematic_subgroups}")
        print(f"Device: {device}")
        print("------------------------------------\n")

    ## Set seed for reproducibility
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    df_train, df_valid, num_labels, label2id, id2label = read_data(
        df_folder=args.df_folder, 
        contrastive_subgroups=False, 
        verbose=False)

    ## Model & Feature Extractor
    model_checkpoint = args.model
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint)
    model = DualModel(
            model_name=model_checkpoint, 
            num_classes=num_labels, 
            num_subgroups=args.num_problematic_subgroups,
            label2id=label2id,
            id2label=id2label,
            hidden_size=768 if 'base' in model_checkpoint else 1024,
            local_files_only=False)
    model.to(device)
    if args.verbose:
        print("Model loaded successfully!")
        print("------------------------------------\n")

    ## Define optimizer and warmup scheduler, with 500 warmup steps
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    scheduler_warmup = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=num_wu, T_mult=1, eta_min=0, last_epoch=-1)

    ## Loss function
    loss_fct = nn.CrossEntropyLoss()
    loss_subgroups_fct = nn.BCELoss()

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
        
        ## Train & Valid Datasets 
        max_duration = args.max_duration    # 10.0 for ITALIC, 4.0 for FSC

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
            
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=args.batch, 
            shuffle=True, 
            num_workers=4)
        valid_dataloader = torch.utils.data.DataLoader(
            valid_dataset, 
            batch_size=args.batch, 
            shuffle=False, 
            num_workers=4)

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

        for step, batch in enumerate(tqdm(train_dataloader)):
            if step > num_steps:
                break
            model.train()
            
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch['labels']
            del batch['labels']

            ## Save subgID and labels
            if not first_run:
                subgIDs = batch['subgID']
                del batch['subgID']
                
            ## Forward pass
            output_classification, output_subgroups = model(batch)

            loss_classification = loss_fct(output_classification, labels.long())

            if not first_run:

                ## map subgIDs and output_subgroups to 0-1
                subgIDs[subgIDs > 0.5] = 1
                subgIDs[subgIDs <= 0.5] = 0
                output_subgroups[output_subgroups > 0.5] = 1
                output_subgroups[output_subgroups <= 0.5] = 0

                loss_subgroups = loss_subgroups_fct(
                    output_subgroups.squeeze().float().to(device), 
                    subgIDs.long().float().to(device)
                    )
                loss = loss_classification + loss_subgroups/100
            else:
                loss = loss_classification

            loss.backward()
            if step % num_gas == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler_warmup.step()
                scheduler.step()

            if step % 100 == 0:
                print(f"Step {step} - Loss: {loss.item()}")

        ## Evaluate
        if args.verbose:
            print("------------------------------------")
            print(f"Evaluating the model at epoch {epoch+1}...")
            print("------------------------------------\n")

        model.eval()
        preds_intents_val = []
        labels_val = []

        for step, batch in enumerate(tqdm(valid_dataloader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels_batch = batch['labels']
            del batch['labels']
            with torch.no_grad():
                output_classification, output_subgroups = model(batch)
                preds_intents_val.extend(output_classification.argmax(dim=-1).tolist())
                labels_val.extend(labels_batch.tolist())

        ## Compute metrics
        if args.verbose:
            print("------------------------------------")
            print(f"Computing metrics at epoch {epoch+1}...")
            print("------------------------------------\n")

        ## Accuracy and f1 score
        acc = accuracy_score(labels_val, preds_intents_val)
        f1 = f1_score(labels_val, preds_intents_val, average='macro')
        print(f"Accuracy: {acc}")
        print(f"F1 score: {f1}")

        ## Save predictions
        if args.verbose:
            print("------------------------------------")
            print(f"Saving predictions at epoch {epoch+1}...")
            print("------------------------------------\n")

        df_valid['prediction'] = preds_intents_val == df_valid['label']
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
                print("Subgroups computed successfully!")
                print("----------------------------------\n")

        first_run = False

    ## Save model
    if args.verbose:
        print("------------------------------------")
        print("Saving the model...")
        print("------------------------------------\n")
    
    torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))

    if args.verbose:
        print("Training completed successfully!")
        print("------------------------------------\n")

    ## Evaluate on test set
    if args.verbose:
        print("------------------------------------")
        print("Evaluating on test set...")
        print("------------------------------------\n")

    ## Load test set
    df_test = pd.read_csv(os.path.join(args.df_folder, 'test_data.csv'), index_col=None)
    for index in range(0,len(df_test)):
        df_test.loc[index,'label'] = label2id[df_test.loc[index,'intent']]
    df_test['label'] = df_test['label'].astype(int)

    ## Test Dataset
    if args.verbose:
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
    if args.verbose:
        print("Dataset loaded successfully!")
        print("----------------------------------\n")

    ## Test Dataloader
    if args.verbose:
        print("----------------------------------")
        print("Creating dataloader...")
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=args.batch, 
        shuffle=False, 
        num_workers=4)
    if args.verbose:
        print("Dataloader created successfully!")
        print("----------------------------------\n")

    ## Evaluate
    if args.verbose:
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
    if args.verbose:
        print("------------------------------------")
        print(f"Computing metrics on test set...")
        print("------------------------------------\n")

    ## Accuracy and f1 score
    acc = accuracy_score(labels_test, preds_intents_test)
    f1 = f1_score(labels_test, preds_intents_test, average='macro')
    print(f"Accuracy: {acc}")
    print(f"F1 score: {f1}")

    ## Save predictions
    if args.verbose:
        print("------------------------------------")
        print(f"Saving predictions on test set...")
        print("------------------------------------\n")

    df_test['prediction'] = preds_intents_test == df_test['label']
    df_test['prediction'] = df_test['prediction'].astype(int)
    df_test.to_csv(os.path.join(output_dir, "df_test.csv"), index=False)