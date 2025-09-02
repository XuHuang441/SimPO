# change model_name and model_path to your own model and you are ready to go!
source ~/.bashrc

# Initialize Conda environment
eval "$(conda shell.bash hook)"
conda activate /home/hubing/miniconda3/envs/inpo

model_name="gemma-2-9b-it_off_policy_tdpo_stage_2"
model_path="/home/hubing/SimPO/outputs/gemma-2-9b-it_off_policy_tdpo_stage_2"
# CUDA_VISIBLE_DEVICES=3 python get_alpaca_answer.py --model_name $model_name --model_path $model_path --conv_temp "gemma"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python get_alpaca_answer_fast.py \
    --model_name $model_name \
    --model_path $model_path \
    --conv_temp "gemma" \
    --tensor_parallel_size 8