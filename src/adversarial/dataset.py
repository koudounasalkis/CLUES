import torch
import librosa
import numpy as np
import copy

""" Dataset Class """
class Dataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        examples, 
        feature_extractor, 
        max_duration, 
        contrastive_subgroups=False, 
        fine_grained_clues=False, 
        augmentation=False,
        adversarial=False
        ):
        self.examples = examples['path']
        self.labels = examples['label']
        self.max_duration = max_duration
        self.feature_extractor = feature_extractor
        self.augmentation = augmentation

        ## Features for additional losses (contrastive subgroups, fine-grained clues, adversarial)
        self.contrastive_subgroups = contrastive_subgroups
        self.fine_grained_clues = fine_grained_clues
        self.adversarial = adversarial
        if self.contrastive_subgroups or self.adversarial:
            self.subIDs = examples['subgID']
        if self.fine_grained_clues:
            self.predictions = examples['prediction']

    def __getitem__(self, idx):

        ## Augmentation:
            # 1: Add noise
            # 2: Change speed up
            # 3: Change pitch
            # 4: Change speed down
            # 5: Add noise + Change speed (up) + Change pitch
            # 6: Add noise + Change speed (down) + Change pitch
        if self.augmentation:
            # Augment or not, with a probability of 0.15
            augment = np.random.choice([True, False], p=[0.15, 0.85])
            # Choose augmentation type
            augmentation_type = np.random.choice([1, 2, 3, 4, 5, 6])
            if augment:
                try:
                    audio, sr = librosa.load(self.examples[idx], sr=16_000)
                    if augmentation_type == 1:
                        # Add noise
                        noise = np.random.normal(0, 0.005, audio.shape[0])
                        audio = audio + noise
                    elif augmentation_type == 2:
                        # Change speed up
                        audio = librosa.effects.time_stretch(audio, rate=1.2)
                    elif augmentation_type == 3:
                        # Change pitch
                        audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=4)
                    elif augmentation_type == 4:
                        # Change speed down
                        audio = librosa.effects.time_stretch(audio, rate=0.8)
                    elif augmentation_type == 5:
                        # Add noise + Change speed (up) + Change pitch
                        noise = np.random.normal(0, 0.005, audio.shape[0])
                        audio = audio + noise
                        audio = librosa.effects.time_stretch(audio, rate=1.2)
                        audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=4)
                    elif augmentation_type == 6:
                        # Add noise + Change speed (down) + Change pitch
                        noise = np.random.normal(0, 0.005, audio.shape[0])
                        audio = audio + noise
                        audio = librosa.effects.time_stretch(audio, rate=0.8)
                        audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=4)
                    # Extract features
                    inputs = self.feature_extractor(
                        audio.squeeze(),
                        sampling_rate=self.feature_extractor.sampling_rate, 
                        return_tensors="pt",
                        max_length=int(self.feature_extractor.sampling_rate * self.max_duration), 
                        truncation=True,
                        padding='max_length')
                except:
                    print("Audio not available")

            else:
                try:
                    inputs = self.feature_extractor(
                        librosa.load(self.examples[idx], sr=16_000)[0].squeeze(),
                        sampling_rate=self.feature_extractor.sampling_rate, 
                        return_tensors="pt",
                        max_length=int(self.feature_extractor.sampling_rate * self.max_duration), 
                        truncation=True,
                        padding='max_length'
                        )
                except:
                    print("Audio not available")
        ## No augmentation
        else:
            try:
                inputs = self.feature_extractor(
                    librosa.load(self.examples[idx], sr=16_000)[0].squeeze(),
                    sampling_rate=self.feature_extractor.sampling_rate, 
                    return_tensors="pt",
                    max_length=int(self.feature_extractor.sampling_rate * self.max_duration), 
                    truncation=True,
                    padding='max_length'
                    )
            except:
                print("Audio not available")

        try:
            item = {'input_values': inputs['input_values'].squeeze(0)}
            item["labels"] = torch.tensor(self.labels[idx])
            if self.contrastive_subgroups or self.adversarial:
                item["subgID"] = torch.tensor(self.subIDs[idx])
            if self.fine_grained_clues:
                item["prediction"] = torch.tensor(self.predictions[idx])
        except:
            item = { 'input_values': [], 'labels': [], 'subgID': [], 'prediction': [] }

        return item


    def __len__(self):
        return len(self.examples)