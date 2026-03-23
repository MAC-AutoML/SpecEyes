import os
import json
import glob
import argparse


def normalize_folder(folder):
    return os.path.normpath(folder)


def infer_baseline_folder(folder):
    folder = normalize_folder(folder)
    folder_name = os.path.basename(folder)
    if folder_name == "SpecEyes":
        return folder

    candidate = os.path.join(os.path.dirname(folder), "SpecEyes")
    if os.path.exists(candidate):
        return candidate

    return folder


def infer_judge_results_dir(folder):
    folder = normalize_folder(folder)
    if "eval_results" in folder:
        return folder.replace("eval_results", "judge_results", 1)

    return os.path.join(
        os.path.dirname(folder),
        f"judge_{os.path.basename(folder)}",
    )


def find_model_marker(filename):
    for marker in ["DeepEyes-7B", "DeepEyes-7b", "Thyme-RL", "thyme-rl"]:
        if marker in filename:
            return marker
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', type=str, default='eval_results_deepeyes/SpecEyes', help='Input folder containing latency_summary.json and result files')
    parser.add_argument('--select', type=str, default='all')
    args = parser.parse_args()
    args.input_folder = normalize_folder(args.input_folder)
    
    # Read latency_summary.json
    latency_summary_path = os.path.join(args.input_folder, 'latency_summary.json')
    if not os.path.exists(latency_summary_path):
        print(f"Error: {latency_summary_path} not found!")
        return
    
    with open(latency_summary_path, 'r') as f:
        latency_summary = json.load(f)
    
    # SpecReason results should use SpecEyes baseline from the sibling folder.
    baseline_latency_summary = latency_summary
    baseline_folder = infer_baseline_folder(args.input_folder)
    if baseline_folder != args.input_folder:
        baseline_latency_summary_path = os.path.join(baseline_folder, 'latency_summary.json')

        if os.path.exists(baseline_latency_summary_path):
            print(f"Loading baseline data from sibling SpecEyes folder: {baseline_folder}")
            with open(baseline_latency_summary_path, 'r') as f:
                baseline_latency_summary = json.load(f)
        else:
            print(f"Warning: baseline latency summary not found at {baseline_latency_summary_path}, using current folder's baseline data")
            baseline_latency_summary = latency_summary
    
    print("Latency Summary:")
    for filename, total_latency in latency_summary.items():
        print(f"  {filename}: {total_latency:.4f}")
    
    # Group files by their baseline
    baseline_groups = {}
    
    # First, identify all baseline files from the appropriate latency summary
    baseline_files = [f for f in baseline_latency_summary.keys() if 'baseline' in f]
    
    print(f"\nBaseline files found: {len(baseline_files)}")
    for bf in baseline_files:
        print(f"  - {bf}")
    
    # For each non-baseline file, find its corresponding baseline
    for filename in latency_summary.keys():
        if args.select != 'all' and args.select not in filename:
            continue
        if 'baseline' in filename:
            continue
        
        # Let's analyze the filename pattern
        # Example experiment: "hr_hr_bench_4k_DeepEyes-7B_Qwen3-VL-2B-Instruct_6_min_0.96.jsonl"
        # Example baseline: "hr_hr_bench_4k_DeepEyes-7B_baseline_6_None_None.jsonl"
        model_marker = find_model_marker(filename)
        if model_marker is None:
            print(f"Warning: File format not recognized: {filename}")
            continue
        
        # Split the filename to get the part up to and including the model name
        parts = filename.split(model_marker)
        if len(parts) < 2:
            print(f"Warning: Invalid filename format: {filename}")
            continue
        
        # Get the prefix including the model name
        prefix = parts[0] + model_marker
        
        # Create glob pattern: prefix + '_baseline_*'
        glob_pattern = prefix + '_baseline_*'
        
        # Use glob to find matching baseline files
        matching_baselines = []
        for bf in baseline_files:
            if glob.fnmatch.fnmatch(bf, glob_pattern):
                matching_baselines.append(bf)
        
        if matching_baselines:
            # Use the first matching baseline
            base_candidate = matching_baselines[0]
            if base_candidate not in baseline_groups:
                baseline_groups[base_candidate] = []
            baseline_groups[base_candidate].append(filename)
            print(f"  Matched: {filename} -> {base_candidate}")
        else:
            print(f"Warning: Baseline not found for {filename}")
            print(f"  Tried pattern: {glob_pattern}")
    
    # Calculate speedup, sample count, and average latency for all files
    results = {}
    
    # Helper function to read judge results from corresponding acc.jsonl files
    def get_judge_results(filename, source_folder):
        judge_results_dir = infer_judge_results_dir(source_folder)
        
        # Construct judge_results filename by appending "_acc" before ".jsonl"
        judge_filename = filename.replace(".jsonl", "_acc.jsonl")
        judge_filepath = os.path.join(judge_results_dir, judge_filename)
        
        # Read judge results if file exists
        all_acc = None
        small_ratio = None
        small_tokens = None
        large_tokens = None
        tokens_per_second = None
        yes_cnt = None
        no_cnt = None
        
        if os.path.exists(judge_filepath):
            try:
                with open(judge_filepath, 'r') as f:
                    # Read entire file content
                    content = f.read()
                    if content:
                        # Replace NaN, Infinity, -Infinity with null to make it valid JSON
                        content = content.replace('NaN', 'null').replace('Infinity', 'null').replace('-Infinity', 'null')
                        judge_data = json.loads(content)
                        all_acc = judge_data.get("all_acc", None)
                        small_ratio = judge_data.get("small_ratio", None)
                        small_tokens = judge_data.get("small_tokens", None)
                        large_tokens = judge_data.get("large_tokens", None)
                        tokens_per_second = judge_data.get("tokens_per_second", None)
                        yes_cnt = judge_data.get("yes_cnt", None)
                        no_cnt = judge_data.get("no_cnt", None)
            except Exception as e:
                print(f"Warning: Failed to read judge results from {judge_filepath}: {e}")
        # No need for warning if file doesn't exist (some files might not have judge results)

        beta = None
        alpha = None
        
        if yes_cnt is not None:
            beta = no_cnt / (yes_cnt + no_cnt)
            if beta > 0:
                alpha = small_ratio / beta
            else:
                alpha = small_ratio
        
        return all_acc, small_ratio, small_tokens, large_tokens, tokens_per_second, beta, alpha
    
    # First, process all baseline files and add them to results with 1.0x speedup
    for baseline_filename in baseline_files:
        # Get baseline total latency from the appropriate latency summary
        if baseline_filename in baseline_latency_summary:
            baseline_total_latency = baseline_latency_summary[baseline_filename]
        else:
            print(f"Warning: Baseline file {baseline_filename} not found in baseline latency summary")
            continue
        
        # Read baseline jsonl to get sample count
        baseline_jsonl_path = os.path.join(baseline_folder, baseline_filename)
        if not os.path.exists(baseline_jsonl_path):
            print(f"Warning: Baseline file {baseline_jsonl_path} not found")
            continue
            
        with open(baseline_jsonl_path, 'r') as f:
            baseline_sample_count = len(f.readlines())
        
        if baseline_sample_count == 0:
            print(f"Warning: Baseline file {baseline_filename} has no samples")
            continue
        
        baseline_avg_latency = baseline_total_latency / baseline_sample_count
        
        # Get judge results
        all_acc, small_ratio, small_tokens, large_tokens, tokens_per_second, beta, alpha = get_judge_results(
            baseline_filename,
            baseline_folder,
        )
        
        # Add baseline to results with 1.0x speedup
        results[baseline_filename] = {
            'speedup': 1.0,
            'sample_count': baseline_sample_count,
            'avg_latency': baseline_avg_latency,
            'total_latency': baseline_total_latency,
            'all_acc': all_acc,
            'small_ratio': small_ratio,
            'small_tokens': small_tokens,
            'large_tokens': large_tokens,
            'tokens_per_second': tokens_per_second,
            'beta': beta,
            'alpha': alpha,
        }
    
    # Now process all non-baseline files
    for baseline_filename, experiment_filenames in baseline_groups.items():
        # Check if baseline exists in results (it should if we processed it above)
        if baseline_filename not in results:
            print(f"Warning: Baseline {baseline_filename} not found in results, skipping experiments")
            continue
            
        baseline_avg_latency = results[baseline_filename]['avg_latency']
        
        for experiment_filename in experiment_filenames:
            # Get experiment total latency
            experiment_total_latency = latency_summary[experiment_filename]
            
            # Read experiment jsonl to get sample count
            experiment_jsonl_path = os.path.join(args.input_folder, experiment_filename)
            if not os.path.exists(experiment_jsonl_path):
                print(f"Warning: Experiment file {experiment_jsonl_path} not found")
                continue
                
            with open(experiment_jsonl_path, 'r') as f:
                experiment_sample_count = len(f.readlines())
            
            if experiment_sample_count == 0:
                print(f"Warning: Experiment file {experiment_filename} has no samples")
                continue
            
            experiment_avg_latency = experiment_total_latency / experiment_sample_count
            
            # Calculate speedup (baseline / experiment)
            speedup = baseline_avg_latency / experiment_avg_latency
            
            # Get judge results for experiment
            all_acc, small_ratio, small_tokens, large_tokens, tokens_per_second, beta, alpha = get_judge_results(
                experiment_filename,
                args.input_folder,
            )
            
            # Add experiment to results
            results[experiment_filename] = {
                'speedup': speedup,
                'sample_count': experiment_sample_count,
                'avg_latency': experiment_avg_latency,
                'total_latency': experiment_total_latency,
                'all_acc': all_acc,
                'small_ratio': small_ratio,
                'small_tokens': small_tokens,
                'large_tokens': large_tokens,
                'tokens_per_second': tokens_per_second,
                'beta': beta,
                'alpha': alpha,
            }
    
    print("\n" + "=" * 120)
    print("Detailed Results:")
    print("=" * 120)
    
    for filename, data in results.items():
        # Print filename on its own line
        print(f"\nFile: {filename}")
        print(f"{'-' * len(filename)}")
        
        # Print metrics on the next line with nice formatting
        speedup_str = f"Speedup: {data['speedup']:6.4f}x"
        samples_str = f"Samples: {data['sample_count']:6}"
        latency_str = f"Avg Latency: {data['avg_latency']:6.4f}"
        beta_str = ""
        alpha_str = ""

        # Add accuracy metrics if available
        acc_str = ""
        ratio_str = ""
        small_tokens_str = ""
        large_tokens_str = ""
        tokens_per_second_str = ""

        if data['all_acc'] is not None:
            acc_str = f"All Acc: {data['all_acc']:6.2f}%"
        if data['small_ratio'] is not None:
            ratio_str = f"Small Ratio: {data['small_ratio']:6.1f}%"
        if data['small_tokens'] is not None:
            small_tokens_str = f"Tokens[S]: {data['small_tokens']:4}"
        if data['large_tokens'] is not None:
            large_tokens_str = f"Tokens[L]: {data['large_tokens']:4}"
        if data['tokens_per_second'] is not None:
            tokens_per_second_str = f"Tokens/s: {data['tokens_per_second']:6.4f}"
        if data['beta'] is not None:
            beta_str = f"Beta: {data['beta']*100:2.2f}"
        if data['alpha'] is not None:
            alpha_str = f"Alpha: {data['alpha']:2.2f}"
        
        # Print all metrics in a structured way
        print(f"{samples_str} | {acc_str} | {speedup_str} | {alpha_str} | {beta_str} | {ratio_str} | {large_tokens_str} | {small_tokens_str} | {tokens_per_second_str}")
    
    print("\n" + "=" * 120)
    
    # Print as dictionary
    # print("\nSpeedup Dictionary:")
    # print(json.dumps(speedup_results, indent=4, ensure_ascii=False))

if __name__ == '__main__':
    main()
