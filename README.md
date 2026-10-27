# RLEC

<a href="https://arxiv.org/abs/2610.35059"><img src="https://img.shields.io/badge/arXiv-paper-b31b1b?logo=arxiv&logoColor=white" alt="Paper"></a>&nbsp;&nbsp;<a href="https://huggingface.co/yhx12/RLEC"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-ffc107" alt="Models"></a>

RLEC (Reinforcement Learning with Explicit Confidence Targets) trains one language model to follow a confidence target specified in the prompt and directly decide whether to answer or abstain. It preserves target-dependent reward gaps by disabling within-group standard-deviation normalization, uses the stable reward family

$R_p^{(b)}(\text{correct})=(1-p)^b$<br>
$R_p^{(b)}(\text{incorrect})=-p(1-p)^{b-1}$<br>
$R_p^{(b)}(\text{abstain})=0$

and linearly schedules the stability coefficient `b` from 0 to 1 during training. This keeps the optimal answer condition `q(x) > p` unchanged while improving training stability.

## Files

- `prepare_data.py`: builds the mixed DAPO-Math and modified HotpotQA training/evaluation data.
- `reward.py`: implements answer checking, abstention handling, the RLEC reward, and its linear schedule.
- `train_30b.sh`: Qwen3-30B-A3B-Base training configuration based on verl.
- `evaluate.py`: generates model responses and stores raw evaluation records.
- `metrics.py`: grades raw records and computes accuracy, AOC, and RLEC score together.

## Environment

The following uses `venv` and `pip` while matching the package versions used in our experiments:

```bash
python3.11 -m venv ~/venv/rl
source ~/venv/rl/bin/activate
python -m pip install --upgrade pip

python -m pip install "vllm==0.19.0" "transformers==5.2.*"
python -m pip install "tensordict>=0.8.0,<=0.10.0,!=0.9.0" torchdata hydra-core "ray[default]"
python -m pip install peft datasets accelerate hf-transfer codetiming pylatexenc mathruler dill pybind11 matplotlib
python -m pip install liger-kernel "fastapi==0.136.3" pydantic math_verify
python -m pip install flash-linear-attention causal-conv1d --no-build-isolation
python -m pip install "flash-attn==2.8.3" --no-build-isolation
python -m pip install "quack-kernels==0.4.1" "tilelang==0.1.10" "ml_dtypes==0.5.4" "apache-tvm-ffi==0.1.11" "safetensors==0.7.0"
python -m pip install "TransferQueue==0.1.8" "wandb==0.26.1"
python -m pip install -e .
```

The experiments used a CUDA 12.8/PyTorch 2.10 compatible FlashAttention 2.8.3 wheel. If the source build above is unsuitable for your system, install the matching prebuilt wheel instead.

## Data preparation

Set the three input files and tokenizer, then generate the parquet files:

```bash
export DAPO_JSON=/path/to/DAPO-Math-17k/train.json
export HOTPOT_TRAIN_JSON=/path/to/modified-hotpotqa/train.json
export HOTPOT_TEST_JSON=/path/to/modified-hotpotqa/test.json
export TOKENIZER_PATH=Qwen/Qwen3-30B-A3B-Base
python RLEC/prepare_data.py
```

By default, processed data are written to `RLEC/data/processed`. The released configuration selects 15,000 examples from each source, repeats the mixed training set five times, uses the ordinary binary objective for 10% of samples, and otherwise samples explicit targets from `[0, 0.9]`.

## Training

Start or connect all nodes to one Ray cluster, then run the script once on the head node:

```bash
export MODEL_PATH=Qwen/Qwen3-30B-A3B-Base
export WANDB_API_KEY=your_key  # alternatively: wandb login
NNODES=4 NGPUS_PER_NODE=8 bash RLEC/train_30b.sh
```

Paths and common hyperparameters can be overridden through environment variables in `train_30b.sh`. W&B runtime state and checkpoints default to node-local `/tmp`; set `CHECKPOINT_DIR` to a suitable POSIX path when persistent checkpoints are required.
