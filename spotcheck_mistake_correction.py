"""
Spot check: run 5 mistake_correction examples through both models,
print raw outputs and parse results side by side.
"""
import sys
import os
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(__file__))

from tasks.mistake_correction import MistakeCorrectionTask
from tasks.base import TaskConfig
from models.completion_api import CompletionAPI, LLMConfig

import yaml

N = 5

def load_task():
    with open("configs/mistake_correction.yaml") as f:
        cfg = yaml.safe_load(f)
    config = TaskConfig(
        name=cfg["name"],
        dataset_path=cfg["dataset_path"],
        dataset_name=cfg["dataset_name"],
        training_split=cfg["training_split"],
        test_split=cfg["test_split"],
        system_prompt=cfg["system_prompt"],
        ground_truth_format=cfg["ground_truth_format"],
        few_shot_samples=cfg.get("few_shot_samples"),
        stop=cfg.get("stop"),
    )
    return MistakeCorrectionTask(config)

def make_model(model_name):
    return CompletionAPI(LLMConfig(
        provider="completion_api",
        model=model_name,
        api_key=os.environ.get("TINKER_API_KEY"),
        temperature=0.0,
        max_tokens=512,
        is_chat=True,
    ))

def run_spotcheck(model_name, task, examples):
    print(f"\n{'='*70}")
    print(f"MODEL: {model_name}")
    print(f"{'='*70}")
    model = make_model(model_name)
    for i, ex in enumerate(examples):
        prompt = task.get_system_prompt(ex)
        gt = task.format_ground_truth(ex)
        response = model.generate(messages=[], system_prompt=prompt, stop=task.config.stop)
        parsed = task.parse_response(response)
        has_final_answer = "Final Answer:" in response

        print(f"\n--- Example {i+1} ---")
        print(f"Ground truth: {gt}")
        print(f"Has 'Final Answer:': {has_final_answer}")
        print(f"Parsed value:  {parsed}")
        print(f"Correct: {parsed is not None and abs(float(parsed) - float(gt)) < 1e-6}")
        print(f"Raw response:\n{response[:600]}")
        print()

if __name__ == "__main__":
    print("Loading dataset...")
    task = load_task()
    examples = task.get_test_examples()[:N]

    trained_model = "tinker://b21a5b3c-0734-58c6-a8b7-d990e19eb402:train:0/sampler_weights/final"
    base_model    = "Qwen/Qwen3-4B-Instruct-2507"

    run_spotcheck(trained_model, task, examples)
    run_spotcheck(base_model, task, examples)
