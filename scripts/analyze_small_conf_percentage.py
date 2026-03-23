import os
import json
import numpy as np
import argparse

def analyze_confidence(folder_path):
    
    # Iterate over all files in the folder.
    for filename in os.listdir(folder_path):
        all_values = []
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            try:
                # Read the JSON object from disk.
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().replace('NaN', 'null')
                    data = json.loads(content)
                    # Extract confidence scores for the target mode.
                    if 'conf_infos' in data and 'min' in data['conf_infos']:
                        min_dict = data['conf_infos']['min']
                        # Merge correct and incorrect samples.
                        if 'wrong' in min_dict and isinstance(min_dict['wrong'], list):
                            all_values.extend(min_dict['wrong'])
                        if 'right' in min_dict and isinstance(min_dict['right'], list):
                            all_values.extend(min_dict['right'])
            except Exception as e:
                print(f"Error processing file {filename}: {e}")
    
            # Compute summary statistics.
            min_val = min(all_values)
            max_val = max(all_values)
            percentiles = [90, 80, 70, 60, 50, 40, 30, 20, 10]
            
            # Print the results in a readable format.
            print("=" * 60)
            print(f"File: {filename}")
            print("=" * 60)
            print(f"Total values collected: {len(all_values)}")
            print(f"Minimum: {min_val:.6f}")
            print(f"Maximum: {max_val:.6f}")
            for p in percentiles:
                print(f"{p}% : {np.percentile(all_values, p):.6f}")
            print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze confidence values from JSON files")
    parser.add_argument("--input_folder", type=str, default="judge_results_qwen3vl-2b-Instruct", help="Path to the input folder containing JSON files")
    args = parser.parse_args()
    analyze_confidence(args.input_folder)
