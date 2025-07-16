# upgrade openai before eval

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/hubing/miniconda3/envs/sim

export OPENAI_CLIENT_CONFIG_PATH="configs.yaml"

VERSION=2 # todo

if [ "$VERSION" == "2" ]; then
  echo "Running AlpacaEval 2.0..."
  alpaca_eval evaluate_from_model \
              --model_configs 'gemma-2-9b-it-simpo' \
              --annotators_config 'weighted_alpaca_eval_gpt4_turbo' \
              --fn_metric 'get_length_controlled_winrate' \
#              --is_overwrite_leaderboard 'True'

elif [ "$VERSION" == "1" ]; then
  echo "Running AlpacaEval 1.0..."
  export IS_ALPACA_EVAL_2=False
  alpaca_eval --model_outputs 'llama3_sft.json' \
              --annotators_config 'alpaca_eval_gpt4_turbo_fn'
else
  echo "Usage: $0 [1|2]"
  exit 1
fi
