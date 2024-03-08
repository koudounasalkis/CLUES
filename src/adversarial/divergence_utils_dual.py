from transformers import TrainingArguments, Trainer
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.utils import class_weight
import pandas as pd
from tqdm import tqdm
import os 

from divexplorer.FP_DivergenceExplorer import FP_DivergenceExplorer
from divexplorer.FP_Divergence import FP_Divergence

from dataset import Dataset
from ft_utils import WeightedTrainer, define_training_args, compute_metrics

import warnings
warnings.filterwarnings("ignore")

"""
    Function for discretizing the dataframe
    @input:
        - df: dataframe to discretize
        - bins: number of bins
        - attributes: columns to discretize
        - strategy: strategy for discretization
        - adaptive: flag specifying whether to use adaptive discretization
        - round_v: number of decimals to round
        - min_distinct: minimum number of distinct values to discretize
    @output:
        - df_discretized: dataframe discretized
"""
def discretize(
    dfI,
    bins=3,
    attributes=None,
    strategy="quantile",
    adaptive=True,
    round_v=0,
    min_distinct = 10
    ):
    attributes = dfI.columns if attributes is None else attributes
    
    X_discretized = KBinsDiscretizer_continuos(
        dfI,
        attributes,
        bins=bins,
        strategy=strategy,
        adaptive=adaptive,
        round_v=round_v,
        min_distinct=min_distinct,
    )
    for attribute in dfI.columns:
        if attribute not in X_discretized:
            X_discretized[attribute] = dfI[attribute]
    return X_discretized

def KBinsDiscretizer_continuos(
    dt, 
    attributes=None, 
    bins=3, 
    strategy="quantile", 
    adaptive=True, 
    round_v=0, 
    min_distinct = 10):
    
    def _get_edges(input_col, bins, round_v=0):
        from sklearn.preprocessing import KBinsDiscretizer
        est = KBinsDiscretizer(n_bins=bins, encode="ordinal", strategy=strategy)
        est.fit(input_col)
        edges = [i for i in est.bin_edges_][0]
        edges = [round(i, round_v) for i in edges][1:-1]
        if len(set(edges)) != len(edges):
            edges = [
                edges[i]
                for i in range(0, len(edges))
                if len(edges) - 1 == i or edges[i] != edges[i + 1]
            ]
        return edges

    attributes = dt.columns if attributes is None else attributes
    continuous_attributes = [a for a in attributes if dt.dtypes[a] != object]
    X_discretize = dt[attributes].copy()
    for col in continuous_attributes:
        if len(dt[col].value_counts()) > min_distinct:
            if adaptive:
                msg = None
                found = False
                for increased in range(0, 5):
                    edges = _get_edges(dt[[col]], bins + increased, round_v=round_v)
                    if (len(edges) + 1) != bins:
                        msg = f"Not enough data in the bins for attribute {col}\
                            --> bin size is increased from {bins} to {bins+increased}"
                    else:
                        found = True
                        break
                if found == False:
                    edges = _get_edges(dt[[col]], bins, round_v=round_v)
                    msg = f"Not enough data in the bins & adaptive failed for attribute {col}. \
                        Discretized with lower #of bins ({len(edges)} vs {bins})"
                if msg:
                    import warnings
                    warnings.warn(msg)
            else:
                edges = _get_edges(dt[[col]], bins, round_v=round_v)
            for i in range(0, len(edges)):
                if i == 0:
                    data_idx = dt.loc[dt[col] <= edges[i]].index
                    X_discretize.loc[data_idx, col] = f"<={edges[i]:.{round_v}f}"
                if i == len(edges) - 1:
                    data_idx = dt.loc[dt[col] > edges[i]].index
                    X_discretize.loc[data_idx, col] = f">{edges[i]:.{round_v}f}"

                data_idx = dt.loc[
                    (dt[col] > edges[i - 1]) & (dt[col] <= edges[i])
                ].index
                X_discretize.loc[
                    data_idx, col
                ] = f"({edges[i-1]:.{round_v}f}-{edges[i]:.{round_v}f}]"
            if edges == []:
                import warnings
                msg = f"No discretization is performed for attribute '{col}'. \
                    The attribute {col} is removed. \
                    \nConsider changing the size of the bins or the strategy.'"
                warnings.warn(msg)
                X_discretize.drop(columns=[col], inplace=True)
        else:
            X_discretize[col] = X_discretize[col].astype("object")
    return X_discretize

""" 
    Function for discretizing the dataframe
    @input: 
        - df: dataframe to discretize
        - target_col: prediction column
        - train: flag specifying whether it is a train or test df
    @output: 
        - df_discretized: dataframe discretized
"""
def discretize_df(df, target_col, dataset, approach, train=False, verbose=False): 

    ## Target to be used for discretization
    if train:
        target_col = 'path'
        if verbose:
            print("Discretizing training data...")
    else:
        target_col = 'prediction'
        if verbose:
            print("Discretizing test data...")
        
    if approach == 'divexplorer':
        ## Columns to be used for discretization
        if dataset == 'fsc':
            demo_cols = ['Self-reported fluency level ', 'First Language spoken',
                'Current language used for work/school', 'gender', 'ageRange']
            slot_cols = ['action', 'object', 'location']
            signal_cols = ['total_silence', 'total_duration', 'trimmed_duration', 
                'n_words', 'speed_rate_word', 'speed_rate_word_trimmed'] 
        elif dataset == 'italic':
            demo_cols = ['gender', 'age', 'region', 'nationality', 'lisp', 'education']
            slot_cols = ['environment', 'device', 'field', 'intent']
            signal_cols = ['total_silence', 'total_duration', 'trimmed_duration', 
                'n_words', 'speed_rate_word', 'speed_rate_word_trimmed'] 
        input_cols = demo_cols + slot_cols + signal_cols
            
        ## Discretize dataframe
        df_discretized = discretize(
            df[input_cols+[target_col]],
            bins=3,
            attributes=input_cols,
            strategy="quantile",    
            round_v = 2,
            min_distinct=5)

        ## Replace values with ranges: "low", "medium", "high"
        replace_values = {}
        for i in range(0,len(signal_cols)):
            for v in df_discretized[signal_cols[i]].unique():
                if "<=" == v[0:2]:
                    replace_values[v] = "low"
                elif ">" == v[0]:
                    replace_values[v] = "high"
                elif "("  == v[0] and "]"  == v[-1]:
                    replace_values[v] = "medium"
                else:
                    raise ValueError(v)
            df_discretized[signal_cols[i]].replace(replace_values, inplace=True)
        
        if dataset == 'fsc':
            df_discretized.loc[df_discretized["location"]=="none_location", "location"] = "none"
            df_discretized.loc[df_discretized["object"]=="none_object", "object"] = "none"

    elif approach == 'clustering':
        ## Columns to be used (just the cluster ids)
        input_cols = ['speech_cluster_id']
        df_discretized = df[input_cols+[target_col]+['intent']]

    else:
        raise ValueError(f"Approach {approach} not supported")

    if verbose:
        print("Data discretized!\n")
    return df_discretized
    

""" 
    Function for compute the divergence
    @input: 
        - df_discretized: dataframe discretized
        - target_col: prediction column
        - target_div: target divergence
        - target_metric: target metric
        - approach: approach used for discretization
        - min_sup: minimum support for frequent pattern mining
    @output:
        - FP_div: frequent pattern divergence
"""
def compute_divergence(
    df_discretized, 
    target_col, 
    target_div, 
    target_metric, 
    approach, 
    min_sup=0.03, 
    num_clusters=20, 
    verbose=False
    ):

    if verbose:
        print("Computing Divergence...")

    ## Divergence
    if approach == 'clustering':
        df_discretized_k = df_discretized[['speech_cluster_id'] + [target_col]]
        df_discretized = df_discretized_k.copy()

    fp_diver = FP_DivergenceExplorer(
        df_discretized, 
        true_class_name=target_col, 
        class_map={"P":1, "N":0}
        )
    FP_fm = fp_diver.getFrequentPatternDivergence(min_support=min_sup, metrics=[target_metric])

    ## Columns to be shown  
    show_cols = ['support', 'itemsets', '#errors', '#corrects', 'accuracy', \
                'd_accuracy', 't_value', 'support_count', 'length']
    remapped_cols = {'tn': '#errors', 'tp': '#corrects', 'posr': 'accuracy', \
                target_metric: target_div, 't_value_tp_fn': 't_value'}
    FP_fm.rename(columns=remapped_cols, inplace=True)
    FP_fm = FP_fm[show_cols].copy()

    ## Compute Divergence
    FP_fm['accuracy'] = round(FP_fm['accuracy'], 5)
    FP_fm['d_accuracy'] = round(FP_fm['d_accuracy'], 5)
    FP_fm['t_value'] = round(FP_fm['t_value'], 2)
    fp_divergence = FP_Divergence(FP_fm, target_div)
    FPdiv = fp_divergence.getDivergence(th_redundancy=None)[::-1]

    if verbose:
        print("Divergence computed!\n")

    return FPdiv



""" 
    Define Class Weights: this function computes the weights for each subgroup
    and assign them to the train data 
    @input:
        - folder: folder where the dataset is stored
        - df: dataframe
        - dataset: dataset name
        - approach: approach used for discretization
        - min_sup: 
"""
def retrieving_subgroups(
    folder, 
    df, 
    output_dir, 
    dataset, 
    approach, 
    min_sup=0.03, 
    num_clusters=10, 
    num_problematic_subgroups=2, 
    fine_grained_clues=False,
    feature_extractor=None,
    max_duration=4.0,
    model_checkpoint=None,
    verbose=False
    ):

    if verbose:
        print(approach)
        print(dataset)

    ## DivExplorer targets
    target_col = 'prediction' 
    target_metric = 'd_posr'
    target_div = 'd_accuracy'
    t_value_col = 't_value_tp_fn'

    ## Discretize the dataframe
    df_discretized = discretize_df(
        df=df, 
        target_col=target_col, 
        dataset=dataset, 
        approach=approach, 
        train=False,
        verbose=verbose)

    ## Compute Divergence
    FPdiv = compute_divergence(
        df_discretized, 
        target_col=target_col, 
        target_div=target_div, 
        target_metric=target_metric, 
        approach=approach,
        min_sup=min_sup,
        num_clusters=num_clusters,
        verbose=verbose)

    if verbose:
        print("The worst three subgroups are:")
        print("1: ", FPdiv.head(3).itemsets.values[0], \
            "with divergence: ", FPdiv.head(3)[target_div].values[0]*100, "%")
        print("2: ", FPdiv.head(3).itemsets.values[1], \
            "with divergence: ", FPdiv.head(3)[target_div].values[1]*100, "%")
        print("3: ", FPdiv.head(3).itemsets.values[2], \
            "with divergence: ", FPdiv.head(3)[target_div].values[2]*100, "%\n")
    with open(f"{output_dir}/worst_subgroups_{approach}.txt", "a") as f:
        f.write("The worst three subgroups are:\n")
        f.write("1: " + str(FPdiv.head(3).itemsets.values[0]) \
            + " with divergence: " + str(FPdiv.head(3)[target_div].values[0]*100) + "%\n")
        f.write("2: " + str(FPdiv.head(3).itemsets.values[1]) \
            + " with divergence: " + str(FPdiv.head(3)[target_div].values[1]*100) + "%\n")
        f.write("3: " + str(FPdiv.head(3).itemsets.values[2]) \
            + " with divergence: " + str(FPdiv.head(3)[target_div].values[2]*100) + "%\n\n")

    ## Read and discretize the training data
    df_train = pd.read_csv(os.path.join(folder, "train_data.csv"))
    df_discretized_train = discretize_df(
        df_train, 
        target_col=target_col, 
        dataset=dataset, 
        approach=approach, 
        train=True)

    ## Create a column in the df, and assign a class to each sample:
    # - 1 if the sample is in the most divergent itemset
    # - 2 if the sample is in the second most divergent itemset
    # - 3 if the sample is in the third most divergent itemset
    # - ...
    # - 0 otherwise
    df_train["subgID"] = 0
    itemsets = []
    for i in range(num_problematic_subgroups):
        itemsets.append(list(FPdiv.itemsets.values[i]))
    for i in tqdm(range(0, len(df_discretized_train))):
        for value,itemset in enumerate(itemsets):
            ks = []
            vs = []
            for item in itemset:
                k, v = item.split("=")
                ks.append(k)
                vs.append(v)
            if all(df_discretized_train.loc[i, ks] == vs):
                if df_train.loc[i, "subgID"] == 0:
                    df_train.loc[i, "subgID"] = value+1
                else:
                    continue
            else:
                continue
    ## Map to a binary case, where 1 is the most divergent itemset and 0 otherwise
    df_train.loc[df_train["subgID"]!=0, "subgID"] = 1

    if verbose:
        print("Subgroups retrieved!\n")

    if fine_grained_clues:
        ## Prepare data
        train_dataset = Dataset(
            df_train, 
            feature_extractor, 
            max_duration, 
            contrastive_subgroups=False, 
            fine_grained_clues=False)
        intents = df_train['intent'].unique()
        label2id, id2label = dict(), dict()
        for i, label in enumerate(intents):
            label2id[label] = str(i)
            id2label[str(i)] = label
        num_labels = len(id2label)
        ## Model and Trainer
        model = AutoModelForAudioClassification.from_pretrained(
            model_checkpoint, 
            num_labels=num_labels,
            label2id=label2id,
            id2label=id2label,
            local_files_only=True)
        training_arguments = define_training_args(output_dir='output', batch_size=16)
        trainer = WeightedTrainer(
            model=model,
            args=training_arguments,
            eval_dataset=train_dataset,
            compute_metrics=compute_metrics,
            contrastive_subgroups=False,
            contrastive_intent=False,
            fine_grained_clues=False)
        ## Predictions
        predictions = trainer.predict(train_dataset).predictions
        predictions = np.argmax(predictions, axis=1)
        df_train['predicted_label'] = predictions
        df_train['prediction'] = np.where(df_train['label'] == predictions, 1, 0)
        df_train['prediction'] = df_train['prediction'].astype(int)

    ## Save the new training data
    df_train.to_csv(f"{folder}/new_data/train_data.csv", index=False)

    if verbose:
        print("New training data saved!\n")