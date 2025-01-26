# CLUES

[![License](https://img.shields.io/badge/License-Apache%202.0-red.svg)](LICENSE) 
[![Paper](https://img.shields.io/badge/Paper-Interspeech%202024-blue)]([https://www.interspeech2024.org/](https://www.isca-archive.org/interspeech_2024/koudounas24b_interspeech.pdf))

This repo contains the code for the paper "A Contrastive Learning Approach to Mitigate Bias in Speech Models," which won the Studen Best Paper award at Interspeech 2024.

In this repository, you will find the code to replicate our experiments.  
We do not include the datasets used in the paper as they are publicly available: [FSC](https://fluent.ai/fluent-speech-commands-a-dataset-for-spoken-language-understanding-research/) and [ITALIC](https://huggingface.co/datasets/RiTA-nlp/ITALIC).


## Experimental Settings 

### Datasets 
We evaluate our approach on two publicly available datasets: Fluent Speech Commands (FSC) for the English language and ITALIC for Italian. The FSC dataset comprises 30,043 utterances from 97 speakers, each associated with three labeled slots (action, object, and location). The intent of each utterance is defined as the combination of these slots.
The ITALIC dataset consists of 16,521 audio samples recorded by 70 speakers. Each utterance is annotated with two slots (action and scenario), and the intent is derived from their combination. We use the "Speaker" configuration, wherein the speakers do not overlap in the train, validation, and test splits.

### Metadata
For the above datasets, we consider the following metadata when using DivExplorer to automatically extract subgroups: (i) demographic metadata describing the speaker (e.g., gender, age, language fluency level), (ii) factors related to speaking and recording conditions (e.g., duration of silences, number of words, speaking rate, and noise level), and (iii) intents represented as combinations of action, object, and location for FSC, or action and scenario for ITALIC.  
We discretize continuous metadata using frequency-based discretization into three distinct ranges, labeled as "low", "medium", and "high". 
Hence, continuous values are categorized into discrete bins based on their respective frequencies within the dataset. In the experiments, we explore all subgroups with a minimum frequency $s$ of $0.03$.

### Models
We fine-tune the transformer-based wav2vec 2.0 base (ca. 90M parameters) and multilingual XLSR (ca. 300M parameters) models on the FSC and the ITALIC dataset, respectively. The pre-trained checkpoints of these models are obtained from the [Hugging Face hub](https://huggingface.co/models). We trained the models for $2800$ steps for FSC and $5100$ for ITALIC, using the AdamW optimizer with a learning rate of $10^{-4}$, and $500$ warmup steps. 
By using the multi-similarity loss, we adhere to standard procedures in contrastive learning, selecting positive and negative sample pairs within each batch to optimize model performance. We set the batch size to 32 to ensure consistency and comparability across our analyses. 

### Losses
We introduced three complementary multi-similarity contrastive loss functions targeting different scopes: task, subgroup, and error.
We define our overall training objective as the aggregation of these losses, along with the conventional classification loss $\mathcal{L}_{cls}$ (e.g., cross-entropy), as defined below:
$$\mathcal{L} = \mathcal{L}_{cls} + \lambda_t \mathcal{L}_{t} + \lambda_s \mathcal{L}_{s} + \lambda_e \mathcal{L}_{e}$$
Once normalized the losses within the same range, we set $\lambda_t = 0.3$, $\lambda_s = 0.4$, and $\lambda_e = 0.3$.

### Metrics 
We assess the overall performance of the models based on accuracy and F1 Macro score. 
We also considered the highest negative subgroup divergence ($\Delta^-_{max}$), and we measured the Silhouette w.r.t. the adopted subgroups ($S$), and the Silhouette w.r.t. the correct/incorrect partitions within the subgroups ($S^\pm$).

## License
This code is released under the Apache 2.0 license. See the [LICENSE](LICENSE) file for more details.
