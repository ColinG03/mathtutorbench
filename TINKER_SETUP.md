# Running Benchmarks with Tinker Native Client

## Setup

1. Create a `.env` file with your Tinker API key:
```bash
TINKER_API_KEY=your_api_key_here
```

2. Install dependencies:
```bash
pip install -r requirements.txt
pip install tinker
```

## Running Benchmarks

### Using Checkpoint Path (Recommended)
```bash
uv run python main.py \
  --tasks socratic_questioning.yaml \
  --provider completion_api \
  --model_args model=tinker://177ef5a0-f6c2-5b60-9ad0-a7f44a3bb716:train:0/sampler_weights/kimi_k2_thinking_base_checkpoint,max_tokens=512,is_chat=True,base_url=https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1 \
  --batch_size 10
```

### Using Base Model Name
```bash
uv run python main.py \
  --tasks socratic_questioning.yaml \
  --provider completion_api \
  --model_args model=moonshotai/Kimi-K2-Thinking,max_tokens=512,is_chat=True,base_url=https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1 \
  --batch_size 10
```

Replace the model path/name and task as needed.
