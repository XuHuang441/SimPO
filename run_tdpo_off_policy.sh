# /home/hubing_google_com/miniconda3/envs/sim/bin/pip

set -e

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/hubing_google_com/miniconda3/envs/sim
export PYTHONPATH=$(pwd)

history_paths=()

# divide dataset into 3 subsets with 20000 rows each.
#conda run -n inpo python -m inpo_scripts.split_dataset

# ------------------------iter1------------------------
history_args=""

#  precompute # --config_file ./accelerate_configs/zero2.yaml
#/home/hubing_google_com/miniconda3/envs/sim/bin/accelerate launch --num_processes=8 -m inpo_scripts.precompute_simpo_style \
#     --run_name "inpo_iter1" \
#     --train_dir "/home/hubing_google_com/SimPO/data/gemma2_ufb_part1.jsonl" \
#     --output_dir "data/inpo_iter1/pref" \
#     --ref_model google/gemma-2-9b-it --last_model google/gemma-2-9b-it \
#     --loss_type inpo --lr_scheduler_type cosine \
#     $history_args \
#     --sanity_check False

# train
#echo "iter1: start training"
#
#ACCELERATE_LOG_LEVEL=info /home/hubing_google_com/miniconda3/envs/sim/bin/accelerate launch \
#    --config_file accelerate_configs/deepspeed_zero3.yaml \
#    -m inpo_scripts.run_inpo \
#    training_configs/gemma-2-9b-it-inpo-iter1.yaml \

history_paths+=("/home/hubing_google_com/SimPO/outputs/gemma-2-9b-it_inpo_stage_1/")

#echo "Completed iteration 1"

# ------------------------iter2------------------------
echo "Starting iteration 2"

# precompute
echo "iter2: start precompute"
history_args=""
if [ ${#history_paths[@]} -gt 0 ]; then
    history_args="--history_paths ${history_paths[@]}"
fi
/home/hubing_google_com/miniconda3/envs/sim/bin/accelerate launch --num_processes=8 -m inpo_scripts.precompute_simpo_style \
    --run_name "inpo_iter2" \
    --train_dir "/home/hubing_google_com/SimPO/data/gemma2_ufb_part2.jsonl" \
    --output_dir "/home/hubing_google_com/SimPO/data/inpo_iter2/pref" \
    --ref_model google/gemma-2-9b-it \
    --loss_type inpo --lr_scheduler_type cosine \
    $history_args \
    --sanity_check False

# train
echo "iter2: start training"

ACCELERATE_LOG_LEVEL=info /home/hubing_google_com/miniconda3/envs/sim/bin/accelerate launch \
    --config_file accelerate_configs/deepspeed_zero3.yaml \
    -m inpo_scripts.run_inpo \
    training_configs/gemma-2-9b-it-inpo-iter2.yaml \

history_paths+=("/home/hubing_google_com/SimPO/outputs/gemma-2-9b-it_off_policy_tdpo_stage_2/")

echo "Completed iteration 2"

 #------------------------iter3------------------------

# precompute
#echo "iter3: start precompute"
#history_args=""
#if [ ${#history_paths[@]} -gt 0 ]; then
#    history_args="--history_paths ${history_paths[@]}"
#fi
#/home/hubing_google_com/miniconda3/envs/sim/bin/accelerate launch --num_processes=8 -m inpo_scripts.precompute_simpo_style \
#    --run_name "inpo_iter3" \
#    --train_dir "/home/hubing_google_com/SimPO/data/gemma2_ufb_part3.jsonl" \
#    --output_dir "/home/hubing_google_com/SimPO/data/inpo_iter3/pref" \
#    --ref_model google/gemma-2-9b-it \
#    --loss_type inpo --lr_scheduler_type cosine \
#    $history_args \
#    --sanity_check False
#
## train
#echo "iter3: start training"
#ACCELERATE_LOG_LEVEL=info /home/hubing_google_com/miniconda3/envs/sim/bin/accelerate launch \
#    --config_file accelerate_configs/deepspeed_zero3.yaml \
#    -m inpo_scripts.run_inpo \
#    training_configs/gemma-2-9b-it-inpo-iter3.yaml \
#
#history_paths+=("/home/hubing_google_com/SimPO/outputs/gemma-2-9b-it_inpo_stage_3/")
#
#echo "Completed iteration 3"

              