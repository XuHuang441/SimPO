# /home/zbz5349/anaconda3/envs/sim/bin/pip

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/hubing/miniconda3/envs/sim
export PYTHONPATH=$(pwd)

# ------------------------iter1------------------------
# precompute
conda run -n sim python -m inpo_scripts.precompute \
    --reference_model_path "google/gemma-2-9b-it" \
    --input_dataset_name "princeton-nlp/gemma2-ultrafeedback-armorm" \
    --output_dataset_path "./data/inpo_iter1" \
    --per_device_batch_size 4 \
    --torch_dtype "bfloat16"

# train
# ACCELERATE_LOG_LEVEL=info conda run -n sim accelerate launch \
# --config_file accelerate_configs/deepspeed_zero3.yaml \
# scripts/run_inpo.py \
# training_configs/gemma-2-9b-it-inpo.yaml \
# --set_values model_name_or_path=google/gemma-2-9b-it \
#              dataset_name=./data/gemma2_ufb_part1.jsonl \
#              output_dir=./outputs/gemma-2-9b-it_inpo_stage_1 \
#              run_name=gemma-2-9b-it_inpo_stage_1 \
#              learning_rate=8.0e-7

# ------------------------iter2------------------------
# on policy data gen

# precompute

# train
# ACCELERATE_LOG_LEVEL=info conda run -n sim accelerate launch \
# --config_file accelerate_configs/deepspeed_zero3.yaml \
# scripts/run_inpo.py \
# training_configs/gemma-2-9b-it-inpo.yaml \
# --set_values model_name_or_path=./outputs/gemma-2-9b-it_inpo_stage_1 \
#              dataset_name=./data/gemma2_ufb_part2.jsonl \
#              output_dir=./outputs/gemma-2-9b-it_inpo_stage_2 \
#              run_name=gemma-2-9b-it_inpo_stage_2 \
#              learning_rate=4.0e-7

# ------------------------iter3------------------------
# on policy data gen

# precompute
# conda run -n sim python precompute.py \
#     --reference_model_path "google/gemma-2-9b-it" \
#     --history_model_paths "./outputs/inpo_stage_1" "./outputs/inpo_stage_2" \
#     --input_dataset_name "princeton-nlp/gemma2-ultrafeedback-armorm" \
#     --output_dataset_path "./data/final_precomputed_dataset" \
#     --per_device_batch_size 4 \
#     --torch_dtype "bfloat16"

# train
# ACCELERATE_LOG_LEVEL=info conda run -n sim accelerate launch \
# --config_file accelerate_configs/deepspeed_zero3.yaml \
# scripts/run_inpo.py \
# training_configs/gemma-2-9b-it-inpo.yaml \
# --set_values model_name_or_path=./outputs/gemma-2-9b-it_inpo_stage_2 \
#              dataset_name=./data/gemma2_ufb_part3.jsonl \
#              output_dir=./outputs/gemma-2-9b-it_inpo_stage_3 \
#              run_name=gemma-2-9b-it_inpo_stage_3 \
#              learning_rate=2.0e-7