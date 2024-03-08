export cuda_device=0

##############################################
#                                            #  
#                   FSC                      #
#                                            #             
##############################################

CUDA_VISIBLE_DEVICES=$cuda_device python clustering_analysis.py \
    --feature_extractor facebook/wav2vec2-base \
    --max_duration 4.0 \
    --df_folder results/fsc/original/ \
    --model_name_or_path results/fsc/original/checkpoint-2500 \
    --save_folder results/fsc/original/ \
    --show_labels \
    --approach both \
    --dataset fsc

CUDA_VISIBLE_DEVICES=$cuda_device python clustering_analysis.py \
    --feature_extractor facebook/wav2vec2-base \
    --max_duration 4.0 \
    --df_folder results/fsc/contrastive/ \
    --model_name_or_path results/fsc/contrastive/checkpoint-2500 \
    --save_folder results/fsc/contrastive/ \
    --show_labels \
    --approach both \
    --dataset fsc 
    
CUDA_VISIBLE_DEVICES=$cuda_device python clustering_analysis.py \
    --feature_extractor facebook/wav2vec2-base \
    --max_duration 4.0 \
    --df_folder results/fsc/contrastive_subgroups/ \
    --model_name_or_path results/fsc/contrastive_subgroups/checkpoint-2500 \
    --save_folder results/fsc/contrastive_subgroups/ \
    --show_labels \
    --approach both \
    --dataset fsc 

CUDA_VISIBLE_DEVICES=$cuda_device python clustering_analysis.py \
    --feature_extractor facebook/wav2vec2-base \
    --max_duration 4.0 \
    --df_folder results/fsc/contrastive_subgroups_errors/ \
    --model_name_or_path results/fsc/contrastive_subgroups_errors/checkpoint-2500 \
    --save_folder results/fsc/contrastive_subgroups_errors/ \
    --show_labels \
    --approach both \
    --dataset fsc 

CUDA_VISIBLE_DEVICES=$cuda_device python clustering_analysis.py \
    --feature_extractor facebook/wav2vec2-base \
    --max_duration 4.0 \
    --df_folder results/fsc/clues/ \
    --model_name_or_path results/fsc/clues/checkpoint-2500 \
    --save_folder results/fsc/clues/ \
    --show_labels \
    --approach both \
    --dataset fsc



##############################################
#                                            #  
#                   ITALIC                   #
#                                            #             
##############################################

CUDA_VISIBLE_DEVICES=$cuda_device python clustering_analysis.py \
    --feature_extractor facebook/wav2vec2-xls-r-300m \
    --max_duration 10.0 \
    --df_folder results/italic/original/ \
    --model_name_or_path results/italic/original/checkpoint-5100 \
    --save_folder results/italic/original/ \
    --show_labels \
    --approach both \
    --dataset italic

CUDA_VISIBLE_DEVICES=$cuda_device python clustering_analysis.py \
    --feature_extractor facebook/wav2vec2-xls-r-300m \
    --max_duration 10.0 \
    --df_folder results/italic/contrastive/ \
    --model_name_or_path results/italic/contrastive/checkpoint-5100 \
    --save_folder results/italic/contrastive/ \
    --show_labels \
    --approach both \
    --dataset italic

CUDA_VISIBLE_DEVICES=$cuda_device python clustering_analysis.py \
    --feature_extractor facebook/wav2vec2-xls-r-300m \
    --max_duration 10.0 \
    --df_folder results/italic/contrastive_subgroups/ \
    --model_name_or_path results/italic/contrastive_subgroups/checkpoint-5100 \
    --save_folder results/italic/contrastive_subgroups/ \
    --show_labels \
    --approach both \
    --dataset italic

CUDA_VISIBLE_DEVICES=$cuda_device python clustering_analysis.py \
    --feature_extractor facebook/wav2vec2-xls-r-300m \
    --max_duration 10.0 \
    --df_folder results/italic/contrastive_subgroups_errors/ \
    --model_name_or_path results/italic/contrastive_subgroups_errors/checkpoint-5100 \
    --save_folder results/italic/contrastive_subgroups_errors/ \
    --show_labels \
    --approach both \
    --dataset italic

CUDA_VISIBLE_DEVICES=$cuda_device python clustering_analysis.py \
    --feature_extractor facebook/wav2vec2-xls-r-300m \
    --max_duration 10.0 \
    --df_folder results/italic/clues/ \
    --model_name_or_path results/italic/clues/checkpoint-5100 \
    --save_folder results/italic/clues/ \
    --show_labels \
    --approach both \
    --dataset italic