import os
import json
import numpy as np
import argparse
from tqdm import tqdm
from io import BytesIO
from PIL import Image
import base64
import io
import requests
from openai import OpenAI

parser = argparse.ArgumentParser()
parser.add_argument('--api_key', type=str, default='EMPTY', help='API key')
parser.add_argument('--api_url', type=str, default='http://localhost:23333/v1', help='API URL')
parser.add_argument('--input_folder', type=str, default="eval_results_deepeyes/v7", help='Path to the input folder')
parser.add_argument('--eval_model_name', type=str, default='qwen72b', help='Model name for evaluation')
args = parser.parse_args()

openai_api_key = args.api_key
openai_api_base = args.api_url
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
if args.eval_model_name is None:
    response = requests.get(f"{openai_api_base}/models")
    models = response.json()
    eval_model_name = models['data'][0]['id']
else:
    eval_model_name = args.eval_model_name

test_types = ['direct_attributes', 'relative_position']
per_type_acc = {}
for test_type in test_types:
    per_type_acc[test_type] = []

abc_map = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F'}

def get_chat_template():
    chat_template = """
Below are two answers to a question. Question is [Question], [Standard Answer] is the standard answer to the question, and [Model_answer] is the answer extracted from a model's output to this question.  Determine whether these two answers are consistent.
Note that [Model Answer] is consistent with [Standard Answer] whenever they are essentially the same. If the meaning is expressed in the same way, it is considered consistent, for example, 'pink' and 'it is pink'.
If they are consistent, Judement is 1; if they are different, Judement is 0. Just output Judement and don't output anything else.\n\n
"""
    return chat_template

def get_gpt4_score_ICE():
    example_1 = """
[Question]: Is the countertop tan or blue?
[Standard Answer]: A. The countertop is tan.
[Model_answer] : tan
Judgement: 1
""" # noqa

    example_2 = """
[Question]: On which side of the picture is the barrier?
[Standard Answer]: A. The barrier is on the left side of the picture.
[Model_answer] : A
Judgement: 1
""" # noqa

    example_3 = """
[Question]: Is the kite brown and large?
[Standard Answer]: A. Yes, the kite is brown and large.
[Model_answer] : Yes
Judgement: 1
""" # noqa

    example_4 = """
[Question]: Are the spots on a giraffe?
[Standard Answer]: A. No, the spots are on a banana.
[Model_answer] : no
Judgement: 1
""" # noqa

    example_5 = """
[Question]: Who is wearing pants?
[Standard Answer]: A. The boy is wearing pants.
[Model_answer] : C. The girl in the picture is wearing pants.
Judgement: 0
""" # noqa

    example_6 = """
[Question]: Is the man phone both blue and closed?
[Standard Answer]: A. Yes, the man phone is both blue and closed.
[Model_answer] : No.
Judgement: 0
""" # noqa

    example_7 = """
[Question]: What color is the towel in the center of the picture?
[Standard Answer]: A. The towel in the center of the picture is blue.
[Model_answer] : The towel in the center of the picture is pink.
Judgement: 0
""" # noqa

    return [example_1, example_2, example_3, example_4, example_5, example_6, example_7]

def get_prompt(predict_str, ground_truth, question):
    examples = get_gpt4_score_ICE()
    chat_template = get_chat_template()
    demo_prompt = chat_template
    for example in examples:
        demo_prompt += example + '\n\n'
    test_prompt = f"""
[Question]: {question}
[Standard Answer]: {ground_truth}
[Model_answer] : {predict_str}
Judgement:"""
    full_prompt = f'{demo_prompt}{test_prompt}'


    return full_prompt




def process(line):
    acc_reward = 0.0
    line = line.strip()
    data = json.loads(line)
    status = 'success'

    if data['status'] == 'error':
        return None, 'error', data

    result = data['result']
    answer = result['answer']
    pred_ans = result['pred_ans']
    question = result['question']
    
    answer = 'A. ' + answer

    if '\\boxed' in pred_ans:
        pred_ans = pred_ans.split('\\boxed{')[1].split('}')[0]

    # rule base check
    
    if len(pred_ans)==1:
        if pred_ans == 'A':
            acc_reward = 1.0
        else:
            acc_reward = 0.0
    elif len(pred_ans) == 2 and '.' in pred_ans:
        if 'A' in pred_ans:
            acc_reward = 1.0
        else:
            acc_reward = 0.0
    elif answer in pred_ans:
        acc_reward = 1.0
    else:
        full_prompt = get_prompt(pred_ans, answer, question)

        chat_response = client.chat.completions.create(
            model=eval_model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": full_prompt},
            ],
            temperature=0.0,
        )
        response = chat_response.choices[0].message.content.strip()

        if 'Judgement:' in response:
            response = response.split('Judgement:')[-1].strip()
            if '1' in response:
                acc_reward = 1.0
            elif '0' in response:
                acc_reward = 0.0
            else:
                print(f' [WARNING] resp format error {response=}')
                acc_reward = 0.0
        else:
            if response == '1':
                acc_reward = 1.0
            elif response == '0':
                acc_reward = 0.0
            else:
                print(f' [WARNING] resp format error {response=}')
                acc_reward = 0.0
    # for p_out in pred_output:
    #     if p_out['role'] == 'system' or p_out['role'] == 'user':
    #         continue
    #     p_content = p_out['content']
    #     if type(p_content) == str:
    #         p_content_msg = p_content.strip()
    #     elif type(p_content) == list:
    #         for _p_content in p_content:
    #             if _p_content['type'] == 'text':
    #                 p_content_msg = _p_content['text']


    return acc_reward, status, data

def calculate_decile_quantiles(data_list):
    """
    Compute 10% to 90% quantiles for a numeric list.

    Args:
        data_list: Input numeric list. Empty lists return an empty dict.

    Returns:
        A dict that maps 10, 20, ..., 90 to the corresponding quantile values.
    """
    # Validate the input.
    if not isinstance(data_list, list):
        raise TypeError("Input must be a list.")
    if len(data_list) == 0:
        return {}
    if not all(isinstance(x, (int, float)) for x in data_list):
        raise TypeError("All list elements must be numeric (int or float).")
    
    # Convert to a NumPy array before quantile computation.
    data_array = np.array(data_list)
    quantile_dict = {}
    
    # Compute deciles from 10% to 90%.
    for percentile in range(10, 100, 10):
        # NumPy expects quantiles in [0, 1].
        quantile_value = np.quantile(data_array, percentile / 100.0)
        quantile_dict[percentile] = round(quantile_value, 4)  # Keep a compact summary.
    
    return quantile_dict


if __name__ == '__main__':
    import glob
    
    input_folder = args.input_folder
    if "eval_results" in input_folder:
        output_folder = input_folder.replace("eval_results", "judge_results")
    else:
        output_folder = os.path.join(
            os.path.dirname(input_folder),
            f"judge_{os.path.basename(os.path.normpath(input_folder))}",
        )
    os.makedirs(output_folder, exist_ok=True)
    
    # Collect result files to evaluate.
    result_files = []
    result_files = glob.glob(os.path.join(input_folder, "vstar*.jsonl"))
    
    if not result_files:
        print("No files found to process!")
        raise SystemExit(0)
    
    print(f"Found {len(result_files)} files to process:")
    for f in result_files:
        print(f"  - {f}")
    
    # Process each result file.
    for result_file in result_files:
        file_name = os.path.basename(result_file)
        output_file = os.path.join(output_folder, file_name.replace('.jsonl', '_acc.jsonl'))

        # Skip files that were already processed.
        if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
            print(f"Output file {output_file} already exists and is not empty. Skipping {file_name}.")
            continue

        print(f"\n{'='*60}")
        print(f"Processing file: {file_name}")
        print(f"{'='*60}")
        
        compress = file_name.replace('.jsonl', '')
        dataset = file_name.split('_')[0]
        
        # Infer the test split from the file name when possible.
        test_type = 'unknown'
        for tt in test_types:
            if tt in file_name:
                test_type = tt
                break
        
        # Reset per-file accumulators.
        error_preds = []
        all_acc = []
        latency_infos = {}
        error_nums = 0
        
        save_json = []
        
        try:
            with open(result_file, 'r') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"Failed to read file {result_file}: {e}")
            continue

        confidence_scores = []
        confidence_scores_wrong = []
        confidence_scores_right = []
        conf_infos_dict = None
        use_small = 0
        small_acc = []
        large_acc = []
        small_generated_length = []
        large_generated_length = []
        yes_cnt = 0
        no_cnt = 0

        print(f"Processing {len(lines)} lines...")
        for line in tqdm(lines, desc="Processing"):
            result = process(line)
            if result is not None:
                acc_reward, status, data = result
                if status == 'error':
                    continue
                
                # Standard evaluation stores a scalar confidence score.
                confidence_score = data.get('confidence_score', 0)
                # Small-model analysis stores a confidence dict per metric.
                confidence_score = data.get('confidence_score_infos', confidence_score)
                
                # Handle standard evaluation outputs.
                if not isinstance(confidence_score, dict):
                    # Keep scalar scores for quantile summaries.
                    if confidence_score > 0:
                        confidence_scores.append(confidence_score)
                    if data.get('use_model', "") == 'small':
                        small_acc.append(acc_reward)
                        small_generated_length.append(data.get('generated_length', 0))
                        if acc_reward != 1.0:
                            confidence_scores_wrong.append(confidence_score)
                        else:
                            confidence_scores_right.append(confidence_score)
                    else:
                        large_acc.append(acc_reward)
                        large_generated_length.append(data.get('generated_length', 0))
                # Handle small-model confidence breakdowns.
                else:
                    if conf_infos_dict is None:
                        conf_infos_dict = {k: {
                            "wrong": [],
                            "right": []
                        } for k in confidence_score.keys()}
                    for k, v in confidence_score.items():
                        if acc_reward != 1.0:
                            conf_infos_dict[k]['wrong'].append(v)
                        else:
                            conf_infos_dict[k]['right'].append(v)


                if data.get('judge_tc', "") == 'yes':
                    yes_cnt += 1
                elif data.get('judge_tc', "") == 'no':
                    no_cnt += 1

                acc = acc_reward
                all_acc.append(acc)
                
                if acc_reward != 1.0:
                    error_preds.append({
                        'result': data.get('result', ''),
                        'spec_reason_info': data.get('spec_reason_info', ''),
                        'print_messages': data.get('print_messages', '')
                    })
                    error_nums += 1
            
        # Derive throughput from the recorded end-to-end latency.
        latency_file = os.path.join(os.path.dirname(result_file), 'latency_summary.json')
        if os.path.exists(latency_file):
            with open(latency_file, 'r') as f:
                latency_infos = json.load(f)
            total_latency = latency_infos.get(file_name, 0)
            if total_latency > 0:
                tokens = sum(small_generated_length) + sum(large_generated_length)
                tokens_per_second = round(tokens / total_latency, 1)
            else:
                tokens_per_second = 0.0
        else:
            tokens_per_second = 0.0

        save_json = {
            'all_acc': np.mean(all_acc) * 100,
            'small_acc': np.mean(small_acc) * 100,
            'large_acc': np.mean(large_acc) * 100,
            'small_tokens': round(np.mean(small_generated_length), 1),
            'large_tokens': round(np.mean(large_generated_length), 1),
            'tokens_per_second': tokens_per_second,
            'all_cnt': len(all_acc),
            'small_cnt': len(small_acc),
            'small_ratio': len(small_acc)/len(all_acc) * 100 if all_acc else 0.0,
            'large_cnt': len(large_acc),
            'yes_cnt': yes_cnt,
            'no_cnt': no_cnt,
            'error_preds': error_preds,
        }
        
        if conf_infos_dict is not None:
            save_json['conf_infos'] = conf_infos_dict
        else:
            save_json['confidence_scores'] = calculate_decile_quantiles(confidence_scores)
            save_json['confidence_scores_wrong'] = confidence_scores_wrong
            save_json['confidence_scores_right'] = confidence_scores_right
        
        # Save the aggregated metrics.
        with open(output_file, 'w') as f:
            f.write(json.dumps(save_json, ensure_ascii=False, indent=4) + '\n')
        print(f"Saved results to: {output_file}")

        
