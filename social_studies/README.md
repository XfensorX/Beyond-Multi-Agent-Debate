```ssh

# Start GPU Job:
srun --gpus=1 --cpus-per-gpu=16 --pty /bin/zsh -i


cd cKAMP/social_studies

# Run the model on a100:
apptainer run --nv \
  -B $HOME/tgi-data:/data \
  -B $HOME/.cache/huggingface:/root/.cache/huggingface \
  tgi.sif \
  --model-id Qwen/Qwen2.5-0.5B-Instruct \
  --port 8000


# run the model on v100:
apptainer run --nv \
  -B $HOME/tgi-data:/data \
  -B $HOME/.cache/huggingface:/root/.cache/huggingface \
  --env USE_FLASH_ATTENTION=False \
  tgi.sif \
  --model-id Qwen/Qwen2.5-0.5B-Instruct \
  --port 8000
  

# Setup the port forwarding
ssh -NL localhost:8000:localhost:8000 node10.l3s                       
  
```

