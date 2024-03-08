import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils import class_weight
from pytorch_metric_learning import losses

from transformers import TrainingArguments, Trainer

""" Trainer Class """
class WeightedTrainer(Trainer):
    def __init__(
        self, 
        contrastive_intent, 
        contrastive_subgroups, 
        fine_grained_clues, 
        adversarial, 
        **kwargs
        ):
        super().__init__(**kwargs)
        self.contrastive_subgroups = contrastive_subgroups
        self.contrastive_intent = contrastive_intent
        self.fine_grained_clues = fine_grained_clues
        self.adversarial = adversarial
    
    def compute_loss(self, model, inputs, return_outputs=False):

        subIDs = inputs.get("subgID")
        if subIDs is not None:
            inputs.pop("subgID")  
        predictions = inputs.get("prediction")
        if predictions is not None:
            inputs.pop("prediction")

        labels = inputs.get("labels").long()
        outputs = model(**inputs)
        logits = outputs.get("logits")

        ## Compute the loss
        if subIDs is not None:

            ### If contrastive_subgroups, apply contrastive loss taking into account the underperforming subgroups as computed by DivExplorer
            if self.contrastive_subgroups:
                ### If fine-grained contrastive subgroups (CLUES), apply two contrastive losses:
                # - one taking into account the underperforming subgroups as computed by DivExplorer
                # - one taking into account the correct and incorrect samples withing subgroups
                if self.fine_grained_clues:
                    # print("Applying Fine-Grained CLUES + Intent-Contrastive Loss + Standard CE Loss")
                    loss_fct = nn.CrossEntropyLoss()
                    loss_1 = loss_fct(
                        logits.view(-1, self.model.config.num_labels), 
                        labels.view(-1)
                        )
                    multi_similarity_loss_fn = losses.MultiSimilarityLoss()
                    sample_embeddings = torch.mean(logits, dim=1)
                    sample_embeddings = sample_embeddings.view(sample_embeddings.shape[0], -1)
                    ## First-Level CLUES: Across-Subgroups
                    loss_2 = multi_similarity_loss_fn(sample_embeddings, subIDs)
                    ## Second-Level CLUES: Within-Subgroups
                    loss_3 = multi_similarity_loss_fn(sample_embeddings, predictions)
                    ## Intent-Contrastive Loss
                    loss_4 = multi_similarity_loss_fn(sample_embeddings, labels)
                    loss_4 = loss_4 * 5.0
                    ## Total Loss
                    lambda_2 = 0.4
                    lambda_3 = 0.3
                    lambda_4 = 0.3  
                    loss = loss_1 + lambda_2*loss_2 + lambda_3*loss_3 + lambda_4*loss_4

                ## If not fine-grained CLUES
                else:
                    ### If CLUES and contrastive, apply two contrastive losses:
                    # - one taking into account the underperforming subgroups as computed by DivExplorer
                    # - one taking into account the labels
                    if self.contrastive_intent:
                        # print("Applying CLUES + Intent-Contrastive Loss + Standard CE Loss")
                        loss_fct = nn.CrossEntropyLoss()
                        loss_1 = loss_fct(
                            logits.view(-1, self.model.config.num_labels), 
                            labels.view(-1)
                            )
                        multi_similarity_loss_fn = losses.MultiSimilarityLoss()
                        sample_embeddings = torch.mean(logits, dim=1)
                        sample_embeddings = sample_embeddings.view(sample_embeddings.shape[0], -1)
                        loss_2 = multi_similarity_loss_fn(sample_embeddings, subIDs)
                        loss_3 = multi_similarity_loss_fn(sample_embeddings, labels)
                        loss_3 = loss_3 * 5.0
                        lambda_2 = 0.6
                        lambda_3 = 0.4
                        loss = loss_1 + lambda_2*loss_2 + lambda_3*loss_3
                        
                    ### If CLUES only, apply contrastive loss taking into account the underperforming subgroups as computed by DivExplorer
                    else:
                        # print("Applying CLUES + Standard CE Loss")
                        loss_fct = nn.CrossEntropyLoss()
                        loss_1 = loss_fct(
                            logits.view(-1, self.model.config.num_labels), 
                            labels.view(-1)
                            )
                        multi_similarity_loss_fn = losses.MultiSimilarityLoss()
                        sample_embeddings = torch.mean(logits, dim=1)
                        sample_embeddings = sample_embeddings.view(sample_embeddings.shape[0], -1) # Reshape sample_embeddings to be (batch_size, embedding_size)
                        loss_2 = multi_similarity_loss_fn(sample_embeddings, subIDs)
                        loss = loss_1 + loss_2 

            ### If adversarial, apply adversarial loss
            elif self.adversarial:
                # print("Applying Adversarial Loss + Standard CE Loss")
                loss_fct = nn.CrossEntropyLoss()
                loss_1 = loss_fct(
                    logits.view(-1, self.model.config.num_labels), 
                    labels.view(-1)
                    )
                ## the adversarial loss tries to maximize the loss of the model on the adversarial examples,
                # i.e., tries to predict whether a sample is from the correct (problematic) or incorrect (non-problematic) subgroup
                ## TODO need a head to predict the subgroups
                loss_2 = loss_fct(
                    logits.view(-1, self.model.config.num_labels),
                    1 - subIDs.view(-1)
                    )
                loss = loss_1 + loss_2
            
            ### If not CLUES, apply either contrastive loss (+ standard cross-entropy loss) or standard cross-entropy loss only
            else: 
                
                ### If contrastive, compute the loss as the sum of the cross-entropy loss and the multi-similarity loss
                if self.contrastive_intent:
                    # print("Applying Intent-Constrastive Loss + Standard CE Loss")
                    loss_fct = nn.CrossEntropyLoss()
                    loss_1 = loss_fct(
                        logits.view(-1, self.model.config.num_labels), 
                        labels.view(-1)
                        )
                    multi_similarity_loss_fn = losses.MultiSimilarityLoss()
                    sample_embeddings = torch.mean(logits, dim=1)
                    sample_embeddings = sample_embeddings.view(sample_embeddings.shape[0], -1) # Reshape sample_embeddings to be (batch_size, embedding_size)
                    loss_2 = multi_similarity_loss_fn(sample_embeddings, labels)
                    loss_2 = loss_2 * 5.0
                    loss = loss_1 + loss_2

                ### If not contrastive, compute the loss as the standard cross-entropy loss
                else:
                    # print("Applying Standard CE Loss")
                    loss_fct = nn.CrossEntropyLoss()
                    loss = loss_fct(
                        logits.view(-1, self.model.config.num_labels), 
                        labels.view(-1)
                        )
        ### If SubIDs are none, apply standard cross-entropy loss
        else:
            # print("Applying Standard CE Loss")
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                logits.view(-1, self.model.config.num_labels), 
                labels.view(-1)
                )

        return (loss, outputs) if return_outputs else loss


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