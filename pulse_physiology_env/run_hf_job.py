import os

from huggingface_hub import HfApi


HF_TOKEN = os.environ.get("HF_TOKEN")
IMAGE = os.environ.get("PULSE_ER_IMAGE", "clashking9999/pulse-er-env:latest")
ARTIFACT_REPO = os.environ.get("PULSE_ER_ARTIFACT_REPO", "clashking9999/pulse-er-grpo-artifacts")
REPO_URL = os.environ.get("PULSE_ER_REPO_URL", "https://github.com/JatinYadav2006/Pulse-ER-env.git")
REPO_REF = os.environ.get("PULSE_ER_REPO_REF", "kumarthegoat")
MODEL_ID = os.environ.get("PULSE_ER_MODEL", "Qwen/Qwen3-0.6B")
SCENARIO_ID = os.environ.get("PULSE_ER_SCENARIO", "polytrauma_demo")

if not HF_TOKEN:
    raise RuntimeError("Set HF_TOKEN in your environment before running this launcher.")

api = HfApi(token=HF_TOKEN)

job = api.run_job(
    image=IMAGE,
    flavor="a10g-small",
    timeout=60 * 60 * 6,
    secrets={
        "HF_TOKEN": HF_TOKEN,
    },
    env={
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "TRL_EXPERIMENTAL_SILENCE": "1",
        "PULSE_ER_REPO_URL": REPO_URL,
        "PULSE_ER_REPO_REF": REPO_REF,
        "PULSE_ER_ARTIFACT_REPO": ARTIFACT_REPO,
        "PULSE_ER_MODEL": MODEL_ID,
        "PULSE_ER_SCENARIO": SCENARIO_ID,
    },
    command=[
        "bash",
        "-lc",
        r"""
set -euo pipefail

echo "Step 1: install git so the job can pull the latest repo code"
apt-get update
apt-get install -y --no-install-recommends git
rm -rf /var/lib/apt/lists/*

echo "Step 2: clone latest project code"
WORKDIR=/workspace/Pulse-ER-env
git clone --branch "$PULSE_ER_REPO_REF" --single-branch "$PULSE_ER_REPO_URL" "$WORKDIR"
cd "$WORKDIR"

echo "Step 3: patch async OpenEnv calls in trl_env.py if the remote branch is stale"
python - <<'PY'
from pathlib import Path

path = Path("/workspace/Pulse-ER-env/pulse_physiology_env/trl_env.py")
text = path.read_text(encoding="utf-8")

if "_run_client_call" not in text:
    if "import asyncio" not in text:
        text = text.replace(
            "from __future__ import annotations\n\n",
            "from __future__ import annotations\n\nimport asyncio\n\n",
            1,
        )

    marker = (
        "        self.reward = 0.0\n"
        "        self.done = False\n"
        "        self.last_observation: PulsePhysiologyObservation | None = None\n"
        "        self.last_tool_result: str | None = None\n"
    )
    helper = (
        marker
        + "\n"
        + "    @staticmethod\n"
        + "    def _run_client_call(awaitable):\n"
        + '        """Bridge the async OpenEnv client into the sync TRL environment API."""\n'
        + "\n"
        + "        try:\n"
        + "            return asyncio.run(awaitable)\n"
        + "        except RuntimeError as exc:\n"
        + '            if "asyncio.run() cannot be called from a running event loop" not in str(exc):\n'
        + "                raise\n"
        + "            loop = asyncio.new_event_loop()\n"
        + "            try:\n"
        + "                return loop.run_until_complete(awaitable)\n"
        + "            finally:\n"
        + "                loop.close()\n"
    )
    text = text.replace(marker, helper, 1)
    text = text.replace(
        "        result = self.client.reset(scenario_id=scenario_id, **reset_kwargs)\n",
        "        result = self._run_client_call(self.client.reset(scenario_id=scenario_id, **reset_kwargs))\n",
        1,
    )
    text = text.replace(
        "        result = self.client.step(action)\n",
        "        result = self._run_client_call(self.client.step(action))\n",
        1,
    )
    path.write_text(text, encoding="utf-8")
    print("Applied async client hotfix to trl_env.py")
else:
    print("trl_env.py already includes async client bridge")
PY

echo "Step 4: install GRPO/training dependencies from fresh repo code"
python -m pip install --no-cache-dir -e "$WORKDIR/pulse_physiology_env[training]" matplotlib jmespath

echo "Step 5: start Pulse server from latest repo code"
python -m uvicorn pulse_physiology_env.server.app:app --host 127.0.0.1 --port 8000 >/tmp/pulse_server.log 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" >/dev/null 2>&1 || true' EXIT

echo "Step 6: wait for server health"
python - <<'PY'
import json
import sys
import time
import urllib.request

for _ in range(90):
    try:
        with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if payload.get("status") == "healthy":
            print("server healthy")
            sys.exit(0)
    except Exception:
        time.sleep(2)

print("server failed to boot")
sys.exit(1)
PY

echo "Step 7: start GRPO training"
OUT_DIR="$WORKDIR/outputs/pulse_er_grpo_real"

python -m pulse_physiology_env.train_grpo \
  --backend real \
  --scenario "$PULSE_ER_SCENARIO" \
  --env-url http://127.0.0.1:8000 \
  --model "$PULSE_ER_MODEL" \
  --output-dir "$OUT_DIR" \
  --num-samples 128 \
  --num-generations 4 \
  --per-device-train-batch-size 8 \
  --gradient-accumulation-steps 4 \
  --num-train-epochs 1 \
  --learning-rate 1e-6

echo "Step 8: convert metrics to PNG and make before/after chart"
python - <<'PY'
import json
from pathlib import Path

import matplotlib.pyplot as plt

out_dir = Path("/workspace/Pulse-ER-env/outputs/pulse_er_grpo_real")
metrics_dir = out_dir / "metrics"
metrics_dir.mkdir(parents=True, exist_ok=True)

reward_points = json.loads((metrics_dir / "reward_history.json").read_text(encoding="utf-8"))
loss_points = json.loads((metrics_dir / "loss_history.json").read_text(encoding="utf-8"))

def plot_series(points, title, ylabel, png_path, color):
    plt.figure(figsize=(8, 5))
    if points:
        xs = [p["step"] for p in points]
        ys = [p["value"] for p in points]
        plt.plot(xs, ys, color=color, linewidth=2)
    plt.title(title)
    plt.xlabel("Training step")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(png_path, dpi=160)
    plt.close()

plot_series(
    reward_points,
    "Pulse-ER GRPO Reward Curve",
    "Reward",
    metrics_dir / "reward_curve.png",
    "#198754",
)
plot_series(
    loss_points,
    "Pulse-ER GRPO Loss Curve",
    "Loss",
    metrics_dir / "loss_curve.png",
    "#0d6efd",
)

before_reward = reward_points[0]["value"] if reward_points else 0.0
after_reward = reward_points[-1]["value"] if reward_points else 0.0

plt.figure(figsize=(6, 5))
plt.bar(
    ["Before training", "After training"],
    [before_reward, after_reward],
    color=["#6c757d", "#198754"],
)
plt.title("Pulse-ER Reward Before vs After Training")
plt.ylabel("Reward")
plt.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(metrics_dir / "before_after_reward.png", dpi=160)
plt.close()

summary = {
    "before_reward": before_reward,
    "after_reward": after_reward,
    "reward_delta": after_reward - before_reward,
}
(metrics_dir / "before_after_reward.json").write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8",
)
PY

echo "Step 9: upload artifacts to Hugging Face dataset"
python - <<'PY'
import os

from huggingface_hub import HfApi

repo_id = os.environ["PULSE_ER_ARTIFACT_REPO"]
job_id = os.environ.get("JOB_ID", "manual-run")

api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
api.upload_folder(
    repo_id=repo_id,
    repo_type="dataset",
    folder_path="/workspace/Pulse-ER-env/outputs/pulse_er_grpo_real",
    path_in_repo=f"jobs/{job_id}",
)
print(f"uploaded artifacts to dataset repo: {repo_id}/jobs/{job_id}")
PY

echo "Job complete"
        """,
    ],
)

print("JOB_ID:", job.id)
print(f"Image: {IMAGE}")
print(f"Repo: {REPO_URL}@{REPO_REF}")
print(f"Open artifacts later at: https://huggingface.co/datasets/{ARTIFACT_REPO}")
