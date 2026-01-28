# TODO:

- make meta configuration saved correctly
- review TODOs
- refactor the whole imports and structure
- Analysis Website

---

phoenix = { cmd = "uv run phoenix serve", cwd = ".", help = "Start the phoenix server. Please assert environment
variables are correctly set, e.g.:  PHOENIX_ALLOW_EXTERNAL_RESOURCES=false PHOENIX_WORKING_DIR='./results/phoenix'" }

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

Available LLMs:

```python
available_llms = {
    "interweb": (
        model="gemma3:1b",
    ),
    "LMStudio": (
        model="qwen2.5-0.5b-instruct",
    ),
    "L3S-TGI": (
        model="Qwen/Qwen2.5-0.5B-Instruct",
    ),
}



```

## Arize Phoenix

```shell
uv add arize-phoenix
PHOENIX_ALLOW_EXTERNAL_RESOURCES=false PHOENIX_WORKING_DIR="./results/phoenix" uv run phoenix serve

```

## Initialization of fresh sever repo:

- Copy environment/.env.pascal
- Create venv: uv sync
- Pull TGI Apptainer SIF: uv run task pull_tgi_apptainer

Then locally in two terminals:

- MODEL_ID="Qwen/Qwen2.5-0.5B-Instruct" uv run task slurm_setup
- uv run task slurm standard

```shell
orch start pascal experiment -m -e 'hetero/baseline'
orch start pascal experiment -m -e 'hetero/mad2'
orch start pascal experiment -m -e 'hetero/mad3'
orch start pascal experiment -m -e 'hetero/mad4'


orch start neumann experiment -m -e 'hetero/baseline'
orch start neumann experiment -m -e 'hetero/mad2'
orch start neumann experiment -m -e 'hetero/mad3'
orch start neumann experiment -m -e 'hetero/mad4'
```