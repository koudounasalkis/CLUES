import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForAudioClassification

import warnings
warnings.filterwarnings('ignore')

"""
    Dual model that jointly tackles classification and adversarial tasks
"""
class DualModel(torch.nn.Module):
    def __init__(
        self, 
        model_name, 
        num_classes=31, 
        num_subgroups=50, 
        label2id=None, 
        id2label=None, 
        hidden_size=768, 
        local_files_only=False
        ):
        super(DualModel, self).__init__()
        self.num_classes = num_classes
        self.model_name = model_name
        self.base_model = AutoModel.from_pretrained(
            model_name, 
            num_labels=num_classes, 
            label2id=label2id,
            id2label=id2label,
            local_files_only=local_files_only
            )
        self.hidden_size = hidden_size

        self.cls_intents = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_size, 100), # 768 for base, 1024 for large
            torch.nn.ReLU(), 
            torch.nn.Dropout(p=0.1), 
            torch.nn.Linear(100, num_classes)       # 1: Intent labels
            )
        self.cls_subgroups = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_size, 100), # 768 for base, 1024 for large
            torch.nn.ReLU(), 
            torch.nn.Dropout(p=0.1), 
            torch.nn.Linear(100, 1)                 # 2: Undeperforming or not
            )
        
    def forward(self, x):
        
        output = self.base_model(**x)

        output_intents = self.cls_intents(output.last_hidden_state[:,-1])
        output_subgroups = self.cls_subgroups(output.last_hidden_state[:,-1])

        return output_intents, output_subgroups