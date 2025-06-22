import os
import subprocess
import time
import requests
import threading
import queue
import sys

from lm_eval import evaluator



def setup_benchmark_environment(vllm_path: str = "./vllm") -> bool:
    """
    Set up the benchmarking environment by cloning VLLM repo and downloading dataset.

    Args:
        vllm_path: Path where to clone the VLLM repository

    Returns:
        bool: True if setup successful, False otherwise
    """
    try:
        # Clone VLLM repository if it doesn't exist
        if not os.path.exists(vllm_path):
            print("Cloning VLLM repository...")
            subprocess.run([
                "git", "clone", "--branch", "v0.8.5", "https://github.com/vllm-project/vllm.git", vllm_path
            ], check=True)
            print("✓ VLLM repository cloned successfully")
        else:
            print("✓ VLLM repository already exists")

        # Install dependencies
        print("Installing dependencies...")
        subprocess.run([
            "python3", "-m", "pip", "install", "numpy", "aiohttp", "transformers", "vllm"
        ], check=True)
        print("✓ Dependencies installed successfully")

        return True

    except subprocess.CalledProcessError as e:
        print(f"✗ Error during setup: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False



def start_vllm_server(model_name: str, port: int, vllm_args: list[str]) -> subprocess.Popen:
    print(f"Starting VLLM server for {model_name} on port {port}...")
    
    cmd = [
       "python3", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_name,
        "--port", str(port),
    ] + vllm_args
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Create queues to store output
    stdout_queue = queue.Queue()
    stderr_queue = queue.Queue()
    
    def enqueue_output(pipe, queue):
        for line in iter(pipe.readline, ''):
            queue.put(line)
        pipe.close()
    
    # Start threads to read output
    threading.Thread(target=enqueue_output, args=(process.stdout, stdout_queue), daemon=True).start()
    threading.Thread(target=enqueue_output, args=(process.stderr, stderr_queue), daemon=True).start()
    
    # Wait for server to start while printing output
    start_time = time.time()
    while time.time() - start_time < 180:
        # Print any new stdout
        while not stdout_queue.empty():
            print(stdout_queue.get(), end='')
        
        # Print any new stderr
        while not stderr_queue.empty():
            print(stderr_queue.get(), end='', file=sys.stderr)

        # Check if server is responding
        try:
            response = requests.get(f"http://localhost:{port}/health", timeout=1)
            if response.status_code == 200:
                print(f"✓ Server started successfully for {model_name}")
                break
        except requests.RequestException:
            pass
        
        time.sleep(5)
    
    # Print any remaining output
    while not stdout_queue.empty():
        print(stdout_queue.get(), end='')
    while not stderr_queue.empty():
        print(stderr_queue.get(), end='', file=sys.stderr)
    
    return process


def run_benchmark(
    model_name: str,
    port: int = 9123,
    request_rate: float = 10.0,
    num_prompts: int = 1000,
    vllm_path: str = "./vllm",
    input_len: int = 128,
    output_len: int = 1,
) -> dict:    
    benchmark_script = os.path.join(vllm_path, "benchmarks", "benchmark_serving.py")

    assert os.path.exists(benchmark_script)
    
    cmd = [
        "python3", benchmark_script,
        "--backend", "vllm",
        "--model", model_name,
        "--dataset-name", "random",
        "--request-rate", str(request_rate),
        "--num-prompts", str(num_prompts),
        "--host", "127.0.0.1",
        "--port", str(port),
        "--percentile-metrics", "ttft,tpot,e2el",
        "--metric-percentiles", "90,99",
        "--random-input-len", str(input_len),
        "--random-output-len", str(output_len),
    ]
    
    print(f"Running benchmark for {model_name}...")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1000,
            cwd=os.path.join(vllm_path, "benchmarks")
        )
        
        # Always print stdout and stderr regardless of return code
        print("=== Benchmark Output ===")
        print(result.stdout)
        print("=== Benchmark Errors ===")
        print(result.stderr)
        
        if result.returncode == 0:
            print(f"✓ Benchmark completed for {model_name}")
            return parse_benchmark_output(result.stdout, model_name)
        else:
            print(f"✗ Benchmark failed for {model_name}")
            return {"model": model_name, "error": result.stderr}
            
    except subprocess.TimeoutExpired as e:
        print(f"✗ Benchmark timed out for {model_name}")
        print(f"Error: {str(e)}")
        if hasattr(e, 'stdout'):
            print("=== Partial Output ===")
            print(e.stdout)
        if hasattr(e, 'stderr'):
            print("=== Partial Errors ===")
            print(e.stderr)
        return {"model": model_name, "error": f"Timeout: {str(e)}"}
    except Exception as e:
        print(f"✗ Unexpected error for {model_name}: {e}")
        print(f"Error: {str(e)}")
        if hasattr(e, 'stdout'):
            print("=== Partial Output ===")
            print(e.stdout)
        if hasattr(e, 'stderr'):
            print("=== Partial Errors ===")
            print(e.stderr)
        return {"model": model_name, "error": str(e)}


def parse_benchmark_output(output: str, model_name: str) -> dict:
    """
    Parse benchmark output to extract metrics.
    
    Args:
        output: Raw benchmark output
        model_name: Name of the model
        
    Returns:
        Dict: Parsed metrics
    """
    metrics = {"model": model_name}
    
    # Define mapping of output patterns to metric keys
    metric_patterns = {
        "Successful requests:": ("successful_requests", int),
        "Benchmark duration (s):": ("duration_seconds", float),
        "Total input tokens:": ("total_input_tokens", int),
        "Total generated tokens:": ("total_generated_tokens", int),
        "Request throughput (req/s):": ("throughput_req_per_sec", float),
        "Output token throughput (tok/s):": ("throughput_tokens_per_sec", float),
        "Total Token throughput (tok/s):": ("total_token_throughput", float),
        "Mean TTFT (ms):": ("ttft_mean", float),
        "Median TTFT (ms):": ("ttft_median", float),
        "P90 TTFT (ms):": ("ttft_p90", float),
        "P99 TTFT (ms):": ("ttft_p99", float),
        "Mean TPOT (ms):": ("tpot_mean", float),
        "Median TPOT (ms):": ("tpot_median", float),
        "P90 TPOT (ms):": ("tpot_p90", float),
        "P99 TPOT (ms):": ("tpot_p99", float),
        "Mean E2EL (ms):": ("e2el_mean", float),
        "Median E2EL (ms):": ("e2el_median", float),
        "P90 E2EL (ms):": ("e2el_p90", float),
        "P99 E2EL (ms):": ("e2el_p99", float),
    }
    
    for line in output.strip().split('\n'):
        line = line.strip()
        for pattern, (key, dtype) in metric_patterns.items():
            if pattern in line:
                try:
                    value = dtype(line.split()[-1])
                    metrics[key] = value
                except (ValueError, IndexError):
                    continue
                break
    
    return metrics


def bench_single_model(
    model_name: str,
    port: int,
    request_rate: float,
    num_prompts: int,
    vllm_path: str,
    vllm_args: list[str],
    input_len: int,
    output_len: int,
) -> dict:
    server_process = start_vllm_server(model_name=model_name, port=port, vllm_args=vllm_args)

    res = run_benchmark(
        model_name=model_name,
        port=port,
        vllm_path=vllm_path,
        request_rate=request_rate, 
        num_prompts=num_prompts,
        input_len=input_len,
        output_len=output_len
    )

    if server_process:
        print(f"Stopping server for {model_name}...")
        server_process.terminate()
        server_process.wait()

    return res


def _prepare_args(vllm_args: list[str], model_name: str) -> list[str]:
    """
    Convert vllm_args list to key=value format expected by lm_eval
    For example: ["--max-model-len", "4096"] becomes ["max_model_len=4096"]
    """
    
    formatted_vllm_args = []
    i = 0
    while i < len(vllm_args):
        if vllm_args[i].startswith("--"):
            # Remove the -- prefix and replace hyphens with underscores
            key = vllm_args[i][2:].replace("-", "_")
            if i + 1 < len(vllm_args) and not vllm_args[i + 1].startswith("--"):
                # This is a key-value pair
                value = vllm_args[i + 1]
                formatted_vllm_args.append(f"{key}={value}")
                i += 2
            else:
                # This is a boolean flag
                formatted_vllm_args.append(f"{key}=True")
                i += 1
        else:
            # Skip values that don't start with --
            i += 1
    
    model_args = ["pretrained=" + model_name] + formatted_vllm_args
    
    model_args_str = ",".join(model_args)  
    return model_args_str      


def evaluate_model(
    model_name: str,
    vllm_args: list[str],
    num_fewshot: int = 0, 
    limit: int = 200, 
    batch_size: int = 8
) -> float:
    results = evaluator.simple_evaluate(
        model="vllm",
        model_args=_prepare_args(vllm_args, model_name),
        tasks=["hellaswag"],
        device="cuda",
        num_fewshot=num_fewshot,
        limit=limit,
        batch_size=batch_size,
    )
    
    return results["results"]["hellaswag"]["acc_norm,none"]
