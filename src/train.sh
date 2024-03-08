export cuda_device=0

##############################################
#                                            #  
#                    FSC                     #
#                                            #             
##############################################

CUDA_VISIBLE_DEVICES=$cuda_device python clues/ft_main.py \
    --output_dir results/fsc/original \
    --batch 32 \
    --dataset fsc \
    --model facebook/wav2vec2-base \
    --df_folder data/fsc \
    --steps 500 \
    --epochs 5 \
    --max_duration 4.0 \
    --verbose

CUDA_VISIBLE_DEVICES=$cuda_device python clues/ft_main.py \
    --output_dir results/fsc/contrastive \
    --batch 32 \
    --dataset fsc \
    --model facebook/wav2vec2-base \
    --df_folder data/fsc \
    --steps 500 \
    --epochs 5 \
    --max_duration 4.0 \
    --contrastive_intent \
    --verbose

CUDA_VISIBLE_DEVICES=$cuda_device python clues/ft_main.py \
    --output_dir results/fsc/contrastive_subgroups \
    --batch 32 \
    --dataset fsc \
    --model facebook/wav2vec2-base \
    --df_folder data/fsc \
    --steps 500 \
    --epochs 5 \
    --max_duration 4.0 \
    --contrastive_subgroups \
    --approach divexplorer \
    --verbose

CUDA_VISIBLE_DEVICES=$cuda_device python clues/ft_main.py \
    --output_dir results/fsc/contrastive_subgroups_errors \
    --batch 32 \
    --dataset fsc \
    --model facebook/wav2vec2-base \
    --df_folder data/fsc \
    --steps 500 \
    --epochs 5 \
    --max_duration 4.0 \
    --contrastive_subgroups \
    --fine_grained_clues \
    --approach divexplorer \
    --verbose

CUDA_VISIBLE_DEVICES=$cuda_device python clues/ft_main.py \
    --output_dir results/fsc/clues \
    --batch 32 \
    --dataset fsc \
    --model facebook/wav2vec2-base \
    --df_folder data/fsc \
    --steps 500 \
    --epochs 5 \
    --max_duration 4.0 \
    --contrastive_intent \
    --contrastive_subgroups \
    --fine_grained_clues \
    --approach divexplorer \
    --verbose



##############################################
#                                            #  
#                  ITALIC                    #
#                                            #             
##############################################

CUDA_VISIBLE_DEVICES=$cuda_device python clues/ft_main.py \
    --output_dir results/italic/original \
    --batch 32 \
    --dataset italic \
    --model facebook/wav2vec2-xls-r-300m \
    --df_folder data/italic \
    --steps 850 \
    --epochs 7 \
    --max_duration 10.0 \
    --verbose

CUDA_VISIBLE_DEVICES=$cuda_device python clues/ft_main.py \
    --output_dir results/italic/contrastive \
    --batch 32 \
    --dataset italic \
    --model facebook/wav2vec2-xls-r-300m \
    --df_folder data/italic \
    --steps 850 \
    --epochs 7 \
    --max_duration 10.0 \
    --contrastive_intent \
    --verbose

CUDA_VISIBLE_DEVICES=$cuda_device python clues/ft_main.py \
    --output_dir results/italic/contrastive_subgroups \
    --batch 32 \
    --dataset italic \
    --model facebook/wav2vec2-xls-r-300m \
    --df_folder data/italic \
    --steps 850 \
    --epochs 7 \
    --max_duration 10.0 \
    --contrastive_subgroups \
    --approach divexplorer \
    --verbose

CUDA_VISIBLE_DEVICES=$cuda_device python clues/ft_main.py \
    --output_dir results/italic/contrastive_subgroups_errors \
    --batch 32 \
    --dataset italic \
    --model facebook/wav2vec2-xls-r-300m \
    --df_folder data/italic \
    --steps 850 \
    --epochs 7 \
    --max_duration 10.0 \
    --contrastive_subgroups \
    --fine_grained_clues \
    --approach divexplorer \
    --verbose

CUDA_VISIBLE_DEVICES=$cuda_device python clues/ft_main.py \
    --output_dir results/italic/clues \
    --batch 32 \
    --dataset italic \
    --model facebook/wav2vec2-xls-r-300m \
    --df_folder data/italic \
    --steps 850 \
    --epochs 7 \
    --max_duration 10.0 \
    --contrastive_intent \
    --contrastive_subgroups \
    --fine_grained_clues \
    --approach divexplorer \
    --verbose