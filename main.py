import argparse
import json
from pathlib import Path
import yaml
from typing import Dict, Any

from models.completion_api import create_llm_model, LLMConfig
from tasks.base import TaskConfig
from registry import TaskRegistry
from tqdm import tqdm


def parse_model_args(args_str: str) -> Dict[str, Any]:
    """Parse comma-separated key=value pairs into a dictionary"""
    if not args_str:
        return {}

    args_dict = {}
    for pair in args_str.split(','):
        key, value = pair.split('=')
        # Convert string values to appropriate types
        if value.lower() == 'true':
            value = True
        elif value.lower() == 'false':
            value = False
        elif value.isdigit():
            value = int(value)
        elif value.replace('.', '').isdigit():
            value = float(value)
        args_dict[key] = value
    return args_dict


def load_task_config(config_path: str) -> TaskConfig:
    with open("configs/" + config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return TaskConfig(**config_dict)


def load_checkpoint(checkpoint_file: Path) -> Dict[str, Any]:
    """Load checkpoint if it exists"""
    if checkpoint_file.exists():
        with open(checkpoint_file, 'r') as f:
            return json.load(f)
    return None


def save_checkpoint(checkpoint_file: Path, predictions: list, targets: list, 
                   all_generations: list, completed_indices: set):
    """Save progress to checkpoint file"""
    checkpoint_data = {
        'predictions': predictions,
        'targets': targets,
        'all_generations': all_generations,
        'completed_indices': list(completed_indices)
    }
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Run model evaluation on educational tasks')
    parser.add_argument('--tasks', type=str, required=True,
                        help='Comma-separated list of task config YAML files')
    parser.add_argument("--provider", required=False, choices=['completion_api', 'ollama', 'gemini'],
                        default='ollama', help="LLM provider to use")
    parser.add_argument('--model_args', type=str, required=True,
                        help='Model arguments in key=value format')
    parser.add_argument('--output', type=str, default='results',
                        help='Output directory for results')
    parser.add_argument('--debug', action='store_true',
                      help='Enable debug logging')
    parser.add_argument('--checkpoint_interval', type=int, default=10,
                      help='Save checkpoint every N examples')
    parser.add_argument('--batch_size', type=int, default=5,
                      help='Number of concurrent requests per batch')
    args = parser.parse_args()

    # Parse model arguments
    model_args = parse_model_args(args.model_args)
    model_args['provider'] = args.provider

    # Create config and model
    config = LLMConfig(**model_args)
    model = create_llm_model(config)


    # Process each task
    task_paths = [path.strip() for path in args.tasks.split(',')]
    results = {}

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    for task_path in task_paths:
        # Load task configuration
        task_config = load_task_config(task_path)
        print(task_path)

        # Get task class and initialize
        task_cls = TaskRegistry.get_task(task_config.name)
        task = task_cls(task_config)

        # Setup checkpoint file
        checkpoint_file = output_dir / f"checkpoint-{config.model.split('/')[-1]}-{task_config.name}.json"
        
        # Try to load checkpoint
        checkpoint = load_checkpoint(checkpoint_file)
        if checkpoint:
            print(f"Resuming from checkpoint: {len(checkpoint['predictions'])} examples already completed")
            predictions = checkpoint['predictions']
            targets = checkpoint['targets']
            all_generations = checkpoint['all_generations']
            completed_indices = set(checkpoint['completed_indices'])
        else:
            predictions = []
            targets = []
            all_generations = []
            completed_indices = set()

        # Get all test examples
        test_examples = task.get_test_examples()
        
        # Filter out already completed examples
        remaining_examples = [(idx, ex) for idx, ex in enumerate(test_examples) if idx not in completed_indices]
        
        if not remaining_examples:
            print(f"All examples already completed for {task_config.name}")
        else:
            # Process examples in batches
            batch_size = args.batch_size
            
            for batch_start in tqdm(range(0, len(remaining_examples), batch_size),
                                   desc=f"Processing batches for {task_config.name}",
                                   initial=len(completed_indices),
                                   total=len(test_examples)):
                batch = remaining_examples[batch_start:batch_start + batch_size]
                
                # Prepare batch items
                batch_items = []
                batch_indices = []
                batch_examples = []
                for idx, example in batch:
                    example["shots"] = task_config.few_shot_samples
                    batch_items.append({
                        "messages": [],
                        "system_prompt": task.get_system_prompt(example),
                        "stop": task_config.stop
                    })
                    batch_indices.append(idx)
                    batch_examples.append(example)
                
                try:
                    # Generate batch concurrently (only if model supports it)
                    if hasattr(model, 'generate_batch'):
                        responses = model.generate_batch(batch_items, max_workers=batch_size)
                    else:
                        # Fallback to sequential if batch not supported
                        responses = []
                        for item in batch_items:
                            try:
                                response = model.generate(
                                    messages=item["messages"],
                                    system_prompt=item["system_prompt"],
                                    stop=item.get("stop")
                                )
                                responses.append(response)
                            except Exception as e:
                                print(f"Error in sequential generation: {e}")
                                responses.append(None)
                    
                    # Process results
                    for batch_idx, (original_idx, example) in enumerate(zip(batch_indices, batch_examples)):
                        response = responses[batch_idx]
                        if response is None:
                            print(f"Skipping failed item at index {original_idx}")
                            continue
                        
                        prediction = task.parse_response(response)
                        predictions.append(prediction)
                        print(prediction)
                        formatted_ground_truth = task.format_ground_truth(example)
                        print(formatted_ground_truth)
                        targets.append(formatted_ground_truth)
                        completed_indices.add(original_idx)
                        
                        if "pedagogy" in task_config.name or 'scaffolding' in task_config.name:
                            generation = {
                                "problem": example.get("question", ""),
                                "reference_solution": example.get("reference_solution", "N/A"),
                                "dialog_history": example.get("conversation_json", []),
                                "dialog_formatted": example.get("dialog_history", ""),
                                "ground_truth_response": example.get("ground_truth_response", ""),
                                "generated_teacher_utterance": prediction,
                            }
                            all_generations.append(generation)
                    
                    # Save checkpoint after each batch
                    save_checkpoint(checkpoint_file, predictions, targets, all_generations, completed_indices)
                    
                except Exception as e:
                    print(f"Error processing batch starting at {batch_start}: {e}")
                    print("Saving checkpoint before exiting...")
                    save_checkpoint(checkpoint_file, predictions, targets, all_generations, completed_indices)
                    raise

        # Final checkpoint save
        save_checkpoint(checkpoint_file, predictions, targets, all_generations, completed_indices)
        
        # Save generations file if needed
        if len(all_generations) > 0:
            output_file = output_dir / f"generations-{config.model.split('/')[-1]}-{task_config.name}.json"
            with open(output_file, 'w') as f:
                json.dump(all_generations, f, indent=2)
        
        # Compute metrics
        metrics = task.compute_metrics(predictions, targets)
        results[task_config.name] = metrics
        
        # Clean up checkpoint file on successful completion
        if checkpoint_file.exists():
            checkpoint_file.unlink()
            print(f"Cleaned up checkpoint file: {checkpoint_file}")

    print(results)

    with open(output_dir / f"results-{config.model.split('/')[-1]}.yaml", 'a+') as f:
        yaml.dump(results, f)


if __name__ == "__main__":
    main()