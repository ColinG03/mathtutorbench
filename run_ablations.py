#!/usr/bin/env python3
"""
Parallel ablation runner: runs all 13 checkpoints across all 9 tasks at 10% scale.
After generation, auto-runs reward model scoring on pedagogy/scaffolding outputs.
"""
import subprocess
import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import argparse

# ---------------------------------------------------------------------------
# Checkpoint slug → Tinker model path
# ---------------------------------------------------------------------------
CHECKPOINTS = {
    # Loss function ablation
    "renyi_03_final":       "tinker://4189a4f1-eb22-5afc-982c-15452df314f4:train:0/sampler_weights/final",
    "renyi_05_000022":      "tinker://c43c917a-56ca-5b46-8e86-917ea771bd7b:train:0/sampler_weights/000022",
    "renyi_08_000126":      "tinker://73c7424e-1ced-51fe-ad27-d795d6205757:train:0/sampler_weights/000126",
    "renyi_12_000008":      "tinker://d6176a88-525a-58a2-ab43-691785c0fa81:train:0/sampler_weights/000008",
    "jensen_shannon_final": "tinker://f4a91b15-b3a4-595c-ba7d-18a6333f7ef7:train:0/sampler_weights/final",
    # Model selection ablation
    "model_sel_30b_4b":     "tinker://1454ff7d-0346-5624-98d3-28a69e1e0f1f:train:0/sampler_weights/final",
    "model_sel_235b_4b":    "tinker://0e729c61-2400-5f86-aca7-4baaa6f596b4:train:0/sampler_weights/final",
    # Post-training methods ablation (model_sel_235b_4b = on-policy only)
    "pt_off_policy":        "tinker://5ca58614-a9e8-5652-8b0b-277ce987636a:train:0/sampler_weights/final",
    "pt_sft":               "tinker://7553114f-54b1-5e09-8e1f-f8bd8152b029:train:0/sampler_weights/final",
    "pt_sft_off":           "tinker://9d8cd858-56c2-5629-883b-d58cd4b86153:train:0/sampler_weights/final",
    "pt_sft_on":            "tinker://df6c4b3f-4f66-5932-bad5-12a945001731:train:0/sampler_weights/final",
    "pt_off_on":            "tinker://2454d295-c9f6-52db-8ba4-7958fe7eebf3:train:0/sampler_weights/final",
    "pt_sft_off_on":        "tinker://d509cec3-c7b5-5b78-930a-01d8223ae7a1:train:0/sampler_weights/final",
}

# ---------------------------------------------------------------------------
# All 9 task configs
# ---------------------------------------------------------------------------
ALL_TASKS = [
    "scaffolding_generation.yaml",
    "scaffolding_generation_hard.yaml",
    "pedagogy_following.yaml",
    "pedagogy_following_hard.yaml",
    "problem_solving.yaml",
    "socratic_questioning.yaml",
    "mistake_correction.yaml",
    "mistake_location.yaml",
    "student_solution_correctness.yaml",
]

# Tasks that need reward model scoring after generation
REWARD_MODEL_TASKS = {
    "scaffolding_generation",
    "scaffolding_generation_hard",
    "pedagogy_following",
    "pedagogy_following_hard",
}

TINKER_BASE_URL = "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"


def run_generation(slug: str, model_path: str, tasks: list, max_examples: int,
                   batch_size: int, output_dir: str) -> tuple[str, bool, str]:
    """Run main.py for one checkpoint across all tasks."""
    task_str = ",".join(tasks)
    api_key = os.environ.get("TINKER_API_KEY", "")

    cmd = [
        sys.executable, "main.py",
        "--tasks", task_str,
        "--provider", "completion_api",
        "--model_args", (
            f"model={model_path},"
            f"base_url={TINKER_BASE_URL},"
            f"api_key={api_key},"
            f"is_chat=True"
        ),
        "--output", output_dir,
        "--batch_size", str(batch_size),
        "--max_examples", str(max_examples),
        "--model_name", slug,
    ]

    log_path = Path(output_dir) / f"log-{slug}.txt"
    print(f"[START] {slug}")
    try:
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        with open(log_path, "w", encoding="utf-8") as log_f:
            result = subprocess.run(
                cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
            )
        success = result.returncode == 0
        status = "OK" if success else f"exit {result.returncode}"
        print(f"[{'DONE' if success else 'FAIL'}] {slug} ({status})")
        return slug, success, str(log_path)
    except Exception as e:
        print(f"[ERROR] {slug}: {e}")
        return slug, False, str(e)


def run_reward_model(slug: str, output_dir: str) -> None:
    """Run compute_scaffolding_score.py on all reward-model tasks for one slug."""
    for task_name in REWARD_MODEL_TASKS:
        gen_file = Path(output_dir) / f"generations-{slug}-{task_name}.json"
        if not gen_file.exists():
            print(f"[SKIP RM] {gen_file} not found")
            continue
        print(f"[RM] Scoring {gen_file}")
        cmd = [
            sys.executable,
            "reward_model/compute_scaffolding_score.py",
            "--data_path", str(gen_file),
        ]
        result = subprocess.run(cmd, text=True, capture_output=True)
        if result.returncode != 0:
            print(f"[RM FAIL] {slug}/{task_name}:\n{result.stderr[-500:]}")
        else:
            print(f"[RM DONE] {slug}/{task_name}")


def main():
    parser = argparse.ArgumentParser(description="Run ablation suite across all checkpoints")
    parser.add_argument("--max_examples", type=int, default=115,
                        help="Max examples per task (default 115 ≈ 10%% of ~1150)")
    parser.add_argument("--batch_size", type=int, default=30,
                        help="Concurrent requests per checkpoint (default 30, 13×30=390)")
    parser.add_argument("--output", type=str, default="results",
                        help="Output directory")
    parser.add_argument("--max_workers", type=int, default=13,
                        help="Number of checkpoints to run in parallel")
    parser.add_argument("--skip_reward_model", action="store_true",
                        help="Skip reward model scoring after generation")
    parser.add_argument("--slugs", type=str, default=None,
                        help="Comma-separated subset of slugs to run (default: all 13)")
    args = parser.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)

    # Load .env
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).parent / ".env")
    except ImportError:
        pass

    slugs_to_run = list(CHECKPOINTS.keys())
    if args.slugs:
        slugs_to_run = [s.strip() for s in args.slugs.split(",")]
        unknown = [s for s in slugs_to_run if s not in CHECKPOINTS]
        if unknown:
            print(f"Unknown slugs: {unknown}")
            sys.exit(1)

    print(f"Running {len(slugs_to_run)} checkpoints × {len(ALL_TASKS)} tasks")
    print(f"max_examples={args.max_examples}, batch_size={args.batch_size}, max_workers={args.max_workers}")
    print(f"Slugs: {slugs_to_run}\n")

    failed = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {
            pool.submit(
                run_generation,
                slug,
                CHECKPOINTS[slug],
                ALL_TASKS,
                args.max_examples,
                args.batch_size,
                args.output,
            ): slug
            for slug in slugs_to_run
        }
        for future in as_completed(futures):
            slug, success, info = future.result()
            if not success:
                failed.append(slug)
                print(f"  Log: {info}")

    print("\n--- Generation complete ---")
    if failed:
        print(f"Failed: {failed}")
    else:
        print("All checkpoints succeeded.")

    if not args.skip_reward_model:
        print("\n--- Running reward model scoring ---")
        successful_slugs = [s for s in slugs_to_run if s not in failed]
        for slug in successful_slugs:
            run_reward_model(slug, args.output)
        print("Reward model scoring complete.")


if __name__ == "__main__":
    main()
