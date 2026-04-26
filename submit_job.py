import os
from huggingface_hub import HfApi

api = HfApi()

# First create the repo on HF Hub
api.create_repo(repo_id="JatinYadav2006/Pulse-ER-env", repo_type="model", exist_ok=True)
print("Repo created/verified")

# Get your token
token = api.token

upload_cmd = f"from huggingface_hub import HfApi; api=HfApi(token='{token}'); api.create_repo(repo_id='JatinYadav2006/Pulse-ER-env', repo_type='model', exist_ok=True); api.upload_folder(folder_path='/workspace/outputs', repo_id='JatinYadav2006/Pulse-ER-env', repo_type='model', path_in_repo='training_outputs', token='{token}'); print('Upload done')"

cmd = " && ".join([
    "apt-get update && apt-get install -y git",
    "git clone https://github.com/JatinYadav2006/Pulse-ER-env.git /workspace/pulse-er",
    "cd /workspace/pulse-er",
    "pip install -e ./pulse_physiology_env",
    "pip install trl transformers datasets accelerate jmespath matplotlib",
    "python -m pulse_physiology_env.train_grpo --backend mock --scenario respiratory_distress --model Qwen/Qwen3-0.6B --output-dir /workspace/outputs --num-samples 128 --num-generations 4 --per-device-train-batch-size 8 --gradient-accumulation-steps 4 --num-train-epochs 1 --learning-rate 1e-6",
    f'python -c "{upload_cmd}"',
])

job = api.run_job(
    image='pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel',
    flavor='a10g-small',
    command=['bash', '-lc', cmd],
)
print('Job submitted:', job)
