import os

from huggingface_hub import HfApi, get_token


HF_TOKEN = os.environ.get("HF_TOKEN") or get_token()
IMAGE = os.environ.get("PULSE_ER_IMAGE", "clashking9999/pulse-er-env:latest")
ARTIFACT_REPO = os.environ.get("PULSE_ER_ARTIFACT_REPO", "KChad/Pulse_ER_env")
REPO_URL = os.environ.get("PULSE_ER_REPO_URL", "https://github.com/JatinYadav2006/Pulse-ER-env.git")
REPO_REF = os.environ.get("PULSE_ER_REPO_REF", "kumarthegoat")
MODEL_ID = os.environ.get("PULSE_ER_MODEL", "Qwen/Qwen3-0.6B")
SCENARIO_ID = os.environ.get("PULSE_ER_SCENARIO", "polytrauma_demo")
FLAVOR = os.environ.get("PULSE_ER_FLAVOR", "a10g-small")
TIMEOUT_SECONDS = int(os.environ.get("PULSE_ER_TIMEOUT_SECONDS", str(60 * 60 * 6)))
NUM_SAMPLES = int(os.environ.get("PULSE_ER_NUM_SAMPLES", "128"))
NUM_GENERATIONS = int(os.environ.get("PULSE_ER_NUM_GENERATIONS", "4"))
PER_DEVICE_TRAIN_BATCH_SIZE = int(os.environ.get("PULSE_ER_BATCH_SIZE", "8"))
GRADIENT_ACCUMULATION_STEPS = int(os.environ.get("PULSE_ER_GRAD_ACCUM", "4"))
NUM_TRAIN_EPOCHS = os.environ.get("PULSE_ER_NUM_TRAIN_EPOCHS", "1")
LEARNING_RATE = os.environ.get("PULSE_ER_LEARNING_RATE", "1e-6")
USE_QLORA = os.environ.get("PULSE_ER_USE_QLORA", "1")
LORA_R = os.environ.get("PULSE_ER_LORA_R", "16")
LORA_ALPHA = os.environ.get("PULSE_ER_LORA_ALPHA", "32")
LORA_DROPOUT = os.environ.get("PULSE_ER_LORA_DROPOUT", "0.05")

if not HF_TOKEN:
    raise RuntimeError("Set HF_TOKEN in your environment or run `hf auth login` before running this launcher.")

api = HfApi(token=HF_TOKEN)

job = api.run_job(
    image=IMAGE,
    flavor=FLAVOR,
    timeout=TIMEOUT_SECONDS,
    secrets={
        "HF_TOKEN": HF_TOKEN,
    },
    env={
        "HF_HUB_ENABLE_HF_TRANSFER": "0",
        "TRL_EXPERIMENTAL_SILENCE": "1",
        "PULSE_ER_REPO_URL": REPO_URL,
        "PULSE_ER_REPO_REF": REPO_REF,
        "PULSE_ER_ARTIFACT_REPO": ARTIFACT_REPO,
        "PULSE_ER_MODEL": MODEL_ID,
        "PULSE_ER_SCENARIO": SCENARIO_ID,
        "PULSE_ER_NUM_SAMPLES": str(NUM_SAMPLES),
        "PULSE_ER_NUM_GENERATIONS": str(NUM_GENERATIONS),
        "PULSE_ER_BATCH_SIZE": str(PER_DEVICE_TRAIN_BATCH_SIZE),
        "PULSE_ER_GRAD_ACCUM": str(GRADIENT_ACCUMULATION_STEPS),
        "PULSE_ER_NUM_TRAIN_EPOCHS": str(NUM_TRAIN_EPOCHS),
        "PULSE_ER_LEARNING_RATE": str(LEARNING_RATE),
        "PULSE_ER_USE_QLORA": USE_QLORA,
        "PULSE_ER_LORA_R": str(LORA_R),
        "PULSE_ER_LORA_ALPHA": str(LORA_ALPHA),
        "PULSE_ER_LORA_DROPOUT": str(LORA_DROPOUT),
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

path = Path("/workspace/Pulse-ER-env/trl_env.py")
text = path.read_text(encoding="utf-8")

if "self._loop = asyncio.new_event_loop()" not in text:
    if "import asyncio" not in text:
        text = text.replace(
            "from __future__ import annotations\n\n",
            "from __future__ import annotations\n\nimport asyncio\n\n",
            1,
        )
    if "import threading" not in text:
        text = text.replace("import asyncio\n", "import asyncio\nimport threading\n", 1)

    constructor_marker = (
        "        self.reward = 0.0\n"
        "        self.done = False\n"
        "        self.last_observation: PulsePhysiologyObservation | None = None\n"
        "        self.last_tool_result: str | None = None\n"
    )
    constructor_replacement = (
        constructor_marker
        + "        self._loop = asyncio.new_event_loop()\n"
        + "        self._loop_thread = threading.Thread(target=self._run_loop, daemon=True)\n"
        + "        self._loop_thread.start()\n"
    )
    text = text.replace(constructor_marker, constructor_replacement, 1)

    helper_block = (
        "    def _run_loop(self) -> None:\n"
        + "        # Own a dedicated event loop for the lifetime of this environment.\n"
        + "\n"
        + "        asyncio.set_event_loop(self._loop)\n"
        + "        self._loop.run_forever()\n"
        + "\n"
        + "    async def _call_client_async(self, method_name: str, *args, **kwargs):\n"
        + "        # Execute one async client call on the dedicated event loop.\n"
        + "\n"
        + "        method = getattr(self.client, method_name)\n"
        + "        return await method(*args, **kwargs)\n"
        + "\n"
        + "    def _run_client_call(self, method_name: str, *args, **kwargs):\n"
        + "        # Bridge the async OpenEnv client into the sync TRL environment API.\n"
        + "\n"
        + "        future = asyncio.run_coroutine_threadsafe(\n"
        + "            self._call_client_async(method_name, *args, **kwargs),\n"
        + "            self._loop,\n"
        + "        )\n"
        + "        return future.result()\n"
        + "\n"
        + "    def __del__(self) -> None:\n"
        + "        # Best-effort cleanup for the background event loop and websocket client.\n"
        + "\n"
        + "        loop = getattr(self, \"_loop\", None)\n"
        + "        if loop is None or loop.is_closed():\n"
        + "            return\n"
        + "        try:\n"
        + "            future = asyncio.run_coroutine_threadsafe(self.client.close(), loop)\n"
        + "            future.result(timeout=5)\n"
        + "        except Exception:\n"
        + "            pass\n"
        + "        finally:\n"
        + "            loop.call_soon_threadsafe(loop.stop)\n"
        + "\n"
    )

    if "    @staticmethod\n    def _run_client_call" in text:
        start = text.index("    @staticmethod\n    def _run_client_call")
        end = text.index("    def reset(", start)
        text = text[:start] + helper_block + text[end:]
    elif "    def _run_client_call" in text:
        start = text.index("    def _run_client_call")
        end = text.index("    def reset(", start)
        text = text[:start] + helper_block + text[end:]
    else:
        insert_at = text.index("    def reset(")
        text = text[:insert_at] + helper_block + text[insert_at:]

    text = text.replace(
        "        result = self.client.reset(scenario_id=scenario_id, **reset_kwargs)\n",
        "        result = self._run_client_call(\"reset\", scenario_id=scenario_id, **reset_kwargs)\n",
        1,
    )
    text = text.replace(
        "        result = self._run_client_call(self.client.reset(scenario_id=scenario_id, **reset_kwargs))\n",
        "        result = self._run_client_call(\"reset\", scenario_id=scenario_id, **reset_kwargs)\n",
        1,
    )
    text = text.replace(
        "        result = self.client.step(action)\n",
        "        result = self._run_client_call(\"step\", action)\n",
        1,
    )
    text = text.replace(
        "        result = self._run_client_call(self.client.step(action))\n",
        "        result = self._run_client_call(\"step\", action)\n",
        1,
    )

    path.write_text(text, encoding="utf-8")
    print("Applied dedicated-loop OpenEnv client hotfix to trl_env.py")
else:
    print("trl_env.py already includes dedicated OpenEnv loop bridge")
PY

echo "Step 4: install GRPO/training dependencies from fresh repo code"
python -m pip install --no-cache-dir -e "$WORKDIR[training]" matplotlib jmespath
python -m pip install --no-cache-dir "git+https://github.com/huggingface/transformers.git@main"

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
QLORA_ARGS=()

if [ "$PULSE_ER_USE_QLORA" = "1" ]; then
  QLORA_ARGS=(
    --use-qlora
    --lora-r "$PULSE_ER_LORA_R"
    --lora-alpha "$PULSE_ER_LORA_ALPHA"
    --lora-dropout "$PULSE_ER_LORA_DROPOUT"
  )
fi

python -m pulse_physiology_env.train_grpo \
  --backend real \
  --scenario "$PULSE_ER_SCENARIO" \
  --env-url http://127.0.0.1:8000 \
  --model "$PULSE_ER_MODEL" \
  --output-dir "$OUT_DIR" \
  --num-samples "$PULSE_ER_NUM_SAMPLES" \
  --num-generations "$PULSE_ER_NUM_GENERATIONS" \
  --per-device-train-batch-size "$PULSE_ER_BATCH_SIZE" \
  --gradient-accumulation-steps "$PULSE_ER_GRAD_ACCUM" \
  --num-train-epochs "$PULSE_ER_NUM_TRAIN_EPOCHS" \
  --learning-rate "$PULSE_ER_LEARNING_RATE" \
  "${QLORA_ARGS[@]}"

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
print(
    "Training config: "
    f"samples={NUM_SAMPLES}, generations={NUM_GENERATIONS}, "
    f"batch={PER_DEVICE_TRAIN_BATCH_SIZE}, grad_accum={GRADIENT_ACCUMULATION_STEPS}, "
    f"epochs={NUM_TRAIN_EPOCHS}, lr={LEARNING_RATE}, qlora={USE_QLORA}"
)
print(f"Open artifacts later at: https://huggingface.co/datasets/{ARTIFACT_REPO}")
