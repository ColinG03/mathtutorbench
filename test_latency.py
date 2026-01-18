#!/usr/bin/env python3
"""Test script to compare OpenAI endpoint vs Tinker sampler client latency."""
import time
import os
from dotenv import load_dotenv
from openai import OpenAI
import tinker
from tinker import ServiceClient, types
from tinker_cookbook import renderers
from transformers import AutoTokenizer



# Load environment variables
load_dotenv()

# Models to test
MODELS = [
    # "tinker://177ef5a0-f6c2-5b60-9ad0-a7f44a3bb716:train:0/sampler_weights/kimi_k2_thinking_base_checkpoint",
    # "tinker://6b88e08c-eef6-5a88-9b3f-ff46a955a2e2:train:0/sampler_weights/qwen3_235b_base_checkpoint",
    "tinker://4ff300a7-aef9-564a-be81-2d0d6fcdf811:train:0/sampler_weights/qwen3_8b_rank128_base_checkpoint",
]

BASE_URL = "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"
TINKER_API_KEY = os.getenv("TINKER_API_KEY", "")

# Simple test message
TEST_MESSAGE = "What is 2 + 2? Please answer briefly."

if not TINKER_API_KEY:
    print("Warning: TINKER_API_KEY not found in .env file. Using empty string.")


def test_openai_endpoint(client: OpenAI, model_name: str) -> dict:
    """Test OpenAI endpoint and return timing results."""
    print(f"\n{'='*80}")
    print(f"Testing OpenAI Endpoint - Model: {model_name}")
    print(f"{'='*80}")
    
    start_time = time.time()
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": TEST_MESSAGE}
            ],
            temperature=0.6,
            max_tokens=2048
        )
        end_time = time.time()
        inference_time = end_time - start_time
        
        content = response.choices[0].message.content
        
        result = {
            "method": "OpenAI Endpoint",
            "model": model_name,
            "success": True,
            "inference_time": inference_time,
            "response_length": len(content) if content else 0,
            "response_preview": content[:100] if content else "No response"
        }
        print(f"\n✓ Success! Inference time: {inference_time:.2f} seconds")
        print(f"Response: {content}")
        
    except Exception as e:
        end_time = time.time()
        inference_time = end_time - start_time
        
        result = {
            "method": "OpenAI Endpoint",
            "model": model_name,
            "success": False,
            "inference_time": inference_time,
            "error": str(e)
        }
        print(f"\n✗ Failed! Error: {e}")
        print(f"Time until failure: {inference_time:.2f} seconds")
    
    return result


def test_sampler_client(sampler, tokenizer, renderer, model_name: str, run_number: int = 1) -> dict:
    """Test sampler client and return timing results."""
    run_label = "Cold Start" if run_number == 1 else "Warm"
    print(f"\n{'='*80}")
    print(f"Testing Sampler Client ({run_label}) - Model: {model_name}")
    print(f"{'='*80}")
    
    # Need to manually format
    messages = [{"role": "user", "content": TEST_MESSAGE}]

    # need to use a renderer to apply chat template and think tags, also tokenizes the query
    prompt = renderer.build_generation_prompt(messages)

    params = types.SamplingParams(
        max_tokens=2048, 
        temperature=0.6, 
        stop_sequences=renderer.get_stop_sequences()
    )

    start_time = time.time()
    try:
        future = sampler.sample(prompt=prompt, sampling_params=params, num_samples=1)
        result = future.result()
        
        end_time = time.time()
        inference_time = end_time - start_time
        
        # Extract the generated text
        content = ""
        for sequence in result.sequences:
            content = tokenizer.decode(sequence.tokens)
        
        result = {
            "method": f"Sampler Client ({run_label})",
            "model": model_name,
            "success": True,
            "inference_time": inference_time,
            "response_length": len(content) if content else 0,
            "response_preview": content[:100] if content else "No response"
        }
        print(f"\n✓ Success! Inference time: {inference_time:.2f} seconds")
        print(f"Response: {content}")
        
    except Exception as e:
        end_time = time.time()
        inference_time = end_time - start_time
        
        result = {
            "method": f"Sampler Client ({run_label})",
            "model": model_name,
            "success": False,
            "inference_time": inference_time,
            "error": str(e)
        }
        print(f"\n✗ Failed! Error: {e}")
        print(f"Time until failure: {inference_time:.2f} seconds")
    
    return result


def main():
    """Run latency tests comparing OpenAI endpoint vs sampler client."""
    print("="*80)
    print("API Latency Comparison: OpenAI Endpoint vs Tinker Sampler Client")
    print("="*80)
    print(f"Base URL: {BASE_URL}")
    print(f"API Key: {'*' * 20 if TINKER_API_KEY else 'NOT SET'}")
    print(f"Test message: {TEST_MESSAGE}")
    
    # Create OpenAI client
    openai_client = OpenAI(
        api_key=TINKER_API_KEY if TINKER_API_KEY else "EMPTY",
        base_url=BASE_URL
    )
    
    # Initialize Tinker service client (lightweight)
    print("\n" + "="*80)
    print("Initializing Tinker ServiceClient...")
    print("="*80)
    service_client = ServiceClient()
    
    all_results = []
    
    # Test each model
    for model in MODELS:
        model_short = model.split("/")[-1]
        print(f"\n{'#'*80}")
        print(f"# Testing Model: {model_short}")
        print(f"{'#'*80}")
        
        
        # Test sampler client - first call (cold start)
        print(f"\nCreating sampler client for {model_short}...")
        sampler_start = time.time()
        sampler = service_client.create_sampling_client(model_path=model)
        sampler_init_time = time.time() - sampler_start
        print(f"Sampler client created in {sampler_init_time:.2f} seconds")



        # Get tokenizer for the model
        print(f"\nGetting tokenizer for {model_short}...")
        rest_client = service_client.create_rest_client()
        training_run = rest_client.get_training_run_by_tinker_path(model).result() 
        tokenizer = AutoTokenizer.from_pretrained(training_run.base_model)

        
        renderer = renderers.Qwen3Renderer(tokenizer=tokenizer)


        sampler_cold_result = test_sampler_client(sampler, tokenizer, renderer, model, run_number=1)
        sampler_cold_result["model_short"] = model_short
        sampler_cold_result["sampler_init_time"] = sampler_init_time
        all_results.append(sampler_cold_result)
        time.sleep(1)
        
        # Test sampler client - second call (warm, reusing same sampler)
        sampler_warm_result = test_sampler_client(sampler, tokenizer, renderer, model, run_number=2)
        sampler_warm_result["model_short"] = model_short
        all_results.append(sampler_warm_result)
        time.sleep(1)

        # Test OpenAI endpoint
        openai_result = test_openai_endpoint(openai_client, model)
        openai_result["model_short"] = model_short
        all_results.append(openai_result)
        time.sleep(1)
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"{'Method':<30s} {'Model':<50s} {'Time':>10s} {'Status':<10s}")
    print("-" * 80)
    
    for result in all_results:
        status = "✓" if result["success"] else "✗"
        time_str = f"{result['inference_time']:.2f}s"
        method = result["method"]
        model_name = result["model_short"]
        
        if result["success"]:
            print(f"{method:<30s} {model_name:<50s} {time_str:>10s} {status:<10s}")
        else:
            error_short = result.get('error', 'Unknown')[:30]
            print(f"{method:<30s} {model_name:<50s} {time_str:>10s} {status:<10s} ({error_short})")
    
    # Compare methods for each model
    print("\n" + "="*80)
    print("COMPARISON BY MODEL")
    print("="*80)
    
    for model in MODELS:
        model_short = model.split("/")[-1]
        model_results = [r for r in all_results if r["model_short"] == model_short and r["success"]]
        
        if model_results:
            print(f"\n{model_short}:")
            for result in model_results:
                method = result["method"]
                time_str = result['inference_time']
                if "Sampler Client (Cold Start)" in method and "sampler_init_time" in result:
                    print(f"  {method:<35s}: {time_str:.2f}s (init: {result['sampler_init_time']:.2f}s)")
                else:
                    print(f"  {method:<35s}: {time_str:.2f}s")
            
            # Calculate warm-up gain if we have both cold and warm results
            cold_result = next((r for r in model_results if "Cold Start" in r["method"]), None)
            warm_result = next((r for r in model_results if "Warm" in r["method"]), None)
            
            if cold_result and warm_result:
                warmup_gain = cold_result["inference_time"] - warm_result["inference_time"]
                warmup_percent = (warmup_gain / cold_result["inference_time"]) * 100
                print(f"  Warm-up gain: {warmup_gain:.2f}s ({warmup_percent:.1f}% faster)")


if __name__ == "__main__":
    main()
