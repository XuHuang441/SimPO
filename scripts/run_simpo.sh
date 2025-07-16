source ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/hubing/miniconda3/envs/sim

ACCELERATE_LOG_LEVEL=info /home/hubing/miniconda3/envs/sim/bin/accelerate launch \
--config_file accelerate_configs/deepspeed_zero3.yaml \
scripts/run_simpo.py \
training_configs/gemma-2-9b-it-simpo.yaml
