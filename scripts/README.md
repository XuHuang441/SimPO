# Environment
sim
```shell
conda create -n sim python=3.10 -y
conda activate sim
conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install \
  https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl \
  numpy==1.26.4 \
  accelerate==0.29.2 \
  deepspeed==0.12.2 \
  transformers==4.44.2 \
  trl==0.9.6 \
  huggingface-hub==0.23.2 \
  datasets==2.18.0 \
  peft==0.7.1 \
  wandb


```
inpo
```shell
conda create -n inpo python=3.10 -y
conda activate inpo
pip install \
  torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124 \
  vllm==0.8.5 \
  transformers==4.53.1 \
  datasets==4.0.0 \
  https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl \
  deepspeed==0.17.2 \
  huggingface-hub==0.33.2 \
  flashinfer-python -i https://flashinfer.ai/whl/cu124/torch2.6/

```