import os
import argparse
import json
from tqdm import tqdm
from PIL import Image
import base64
import io
import pandas as pd
from typing import List
from qwen_vl_utils import process_vision_info
import torch
import copy
import time
import traceback
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor, Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoConfig

try:
    from .utils import run_evaluation, smart_resize, encode_pil_image_to_base64, safe_cuda_empty_cache, answer_separability, SYSTEM_PROMPT, generate_prompt_final_qa, execute_code_in_sandbox
    from .prompt import *
except ImportError:
    from utils import run_evaluation, smart_resize, encode_pil_image_to_base64, safe_cuda_empty_cache, answer_separability, SYSTEM_PROMPT, generate_prompt_final_qa, execute_code_in_sandbox
    from prompt import *
import re


ABC_MAP = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F', 7: 'G', 8: 'H', 9: 'I', 10: 'J'}


def parse_arguments():
    parser = argparse.ArgumentParser(description='Thyme Evaluation Script')
    parser.add_argument('--large_model_path', type=str, default="Kwai-Keye/Thyme-RL")
    parser.add_argument('--small_model_path', type=str, default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument('--benchmark', type=str, choices=['vstar', 'hr', 'pope'], default='vstar')
    parser.add_argument('--test_type', type=str, default='all')
    parser.add_argument('--output_path', type=str, default='eval_results_thyme/SpecReason')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument("--score_threshold", type=float, default=7.0, help="Acceptance threshold for utility score")
    # v*
    parser.add_argument('--vstar_path', type=str, default="data/vstar")
    # hr-bench
    parser.add_argument('--hrbench_path', type=str, default="data/HR-Bench")
    # pope
    parser.add_argument('--pope_path', type=str, default="data/POPE")
    args = parser.parse_args()

    return args


def handle_exception(e, error_prefix):
    full_error_info = traceback.format_exc()
    
    print(full_error_info)
    return f"{error_prefix}: {str(e)}"


def build_mc_question_text(question: str, options: List[str]) -> str:
    option_lines = []
    for i, opt in enumerate(options):
        option_lines.append(f"{ABC_MAP[i + 1]}. {opt}")
    option_block = "\n".join(option_lines)
    question_text = (
        f"Question: {question}\nOptions:\n{option_block}\n"
        f"Please select the correct answer from the options above."
    )
    return question_text


def decode_base64_to_temp_image(base64_string: str, save_dir: str, idx: int) -> str:
    ensure_dir(save_dir)
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    if image.mode in ('RGBA', 'P'):
        image = image.convert('RGB')
    out_path = os.path.join(save_dir, f"{idx:08d}.jpg")
    image.save(out_path, format='JPEG', quality=95)
    return out_path


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def init_messages(data_item, args):
    img_path = data_item['img_path']
    question_text = data_item['question_text']
    
    # Load and resize the image for the Thyme prompt.
    pil_img = Image.open(img_path)
    ori_width, ori_height = pil_img.size
    resize_w, resize_h = smart_resize(ori_width, ori_height)
    pil_img_resized = pil_img.resize((resize_w, resize_h), resample=Image.BICUBIC)
    base64_image = encode_pil_image_to_base64(pil_img_resized)
    
    # Build the initial Thyme prompt from the question and image.
    initial_prompt_text = generate_prompt_final_qa(question_text, img_path)
    
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [
            {"type": "image", "image": img_path},
            {"type": "text", "text": initial_prompt_text},
        ]},
    ]

    print_messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [
            {"type": "image", "image": ""},
            {"type": "text", "text": initial_prompt_text},
        ]},
    ]

    return messages, print_messages, initial_prompt_text, base64_image


def get_response(messages, model, processor, image_patch_size=14):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            stop_strings=["</code>", "</answer>"],
            tokenizer=processor.tokenizer,
        )
    if isinstance(outputs, tuple):
        outputs = outputs[0]
    generated_ids_trimmed = outputs[:, inputs.input_ids.shape[1]:]
    output_text = processor.decode(generated_ids_trimmed[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    generated_length = generated_ids_trimmed.shape[1]

    return output_text, generated_length


def get_utility_score(messages, large_model, large_processor):
    """Score the utility of the latest small-model reasoning step."""
    # Build the verification prompt.
    verify_prompt = """
    Please evaluate the quality of the last reasoning step. Score from 0-9, where:
    9: Perfect logic, directly helps solve the problem
    7-8: Good logic, contributes to the solution
    5-6: Neutral, doesn't hurt but doesn't help much
    3-4: Problematic, may lead to confusion
    0-2: Incorrect, misleading, or irrelevant
    
    Only output the score as a single digit.
    """
    
    # Append the verification prompt to the message history.
    verify_messages = copy.deepcopy(messages)
    verify_messages.append({"role": "user", "content": [{"type": "text", "text": verify_prompt}]})
    
    # Generate the utility score with the large model.
    text = large_processor.apply_chat_template(verify_messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(verify_messages)
    inputs = large_processor(text=[text], images=image_inputs, videos=video_inputs, return_tensors="pt").to(large_model.device)
    
    with torch.no_grad():
        outputs = large_model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=large_processor.tokenizer.eos_token_id,
            eos_token_id=large_processor.tokenizer.eos_token_id,
            tokenizer=large_processor.tokenizer,
        )
    
    generated_ids_trimmed = outputs[:, inputs.input_ids.shape[1]:]
    score_text = large_processor.decode(generated_ids_trimmed[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    
    # Extract the first digit from the model output.
    score = 0
    for char in score_text:
        if char.isdigit():
            score = int(char)
            break
    
    return score


def speculative_reasoning(data_item, small_model, small_processor, large_model, large_processor, args):
    """Run the speculate-verify-decide reasoning loop."""
    # Initialize the message state for this sample.
    messages, print_messages, question_prompt, base64_image = init_messages(data_item, args)
    
    status = "success"
    error = ""
    generated_length = 0
    reasoning_history = []
    
    try:
        step_id = 0
        while True:
            if step_id > 5:  # Cap the number of reasoning turns.
                status = "error"
                error = "Step exceeds maximum number of steps"
                break
            
            # Step 1: Speculate using the small model.
            # Rebuild the prompt with accepted reasoning history.
            speculation_messages = copy.deepcopy(messages)
            if reasoning_history:
                history_text = "\n".join(reasoning_history)
                speculation_messages.append({"role": "assistant", "content": [{"type": "text", "text": history_text}]})
            
            # Let the small model propose the next reasoning step.
            small_output, small_length = get_response(speculation_messages, small_model, small_processor)
            generated_length += small_length
            
            # Step 2: Verify the proposed step with the large model.
            # Build the verification prompt from the same history.
            verification_messages = copy.deepcopy(messages)
            if reasoning_history:
                history_text = "\n".join(reasoning_history)
                verification_messages.append({"role": "assistant", "content": [{"type": "text", "text": history_text}]})
            verification_messages.append({"role": "assistant", "content": [{"type": "text", "text": small_output}]})
            
            # Ask the large model for a utility score.
            utility_score = get_utility_score(verification_messages, large_model, large_processor)
            
            # Step 3: Accept or replace the proposed step.
            if utility_score >= args.score_threshold:
                # Accept the small-model step.
                reasoning_history.append(small_output)
                print_messages.append({"role": "assistant", "content": [{"type": "text", "text": small_output}]})
                print_messages.append({"role": "user", "content": [{"type": "text", "text": f"[VERIFIED: Score={utility_score}]"}]})
            else:
                # Reject the step and let the large model replace it.
                # The large model still keeps the ability to run code tools.
                correction_messages = copy.deepcopy(messages)
                if reasoning_history:
                    history_text = "\n".join(reasoning_history)
                    correction_messages.append({"role": "assistant", "content": [{"type": "text", "text": history_text}]})
                
                # Generate a single replacement step.
                output_text, length = get_response(correction_messages, large_model, large_processor)
                large_output = output_text
                large_length = length
                
                # Execute generated code blocks inside the sandbox when present.
                code_regex = re.compile(r'<code>\s*(?:```\s*)?(?:python\s*)?([\s\S]*?)\s*(?:```\s*)?</code>', re.IGNORECASE)
                code_match = code_regex.search(output_text)
                
                if code_match:
                    code_to_execute = code_match.group(1).strip()
                    # Execute the extracted Python code.
                    processed_img_paths, captured_stdout, error_msg, current_execution_context = execute_code_in_sandbox(
                        code_to_execute, data_item['img_path'],
                        previous_execution_context={}
                    )
                    
                    # Attach generated sandbox outputs to the visible trace.
                    if processed_img_paths:
                        # Surface produced images back to the model trace.
                        output_text += "\n<sandbox_output>"
                        for img_path in processed_img_paths:
                            if os.path.exists(img_path):
                                output_text += f"\n![Image]({img_path})"
                        output_text += "\n</sandbox_output>"
                
                # Append the replacement step to the working history.
                correction_messages.append({"role": "assistant", "content": [{"type": "text", "text": output_text}]})
                
                generated_length += large_length
                
                # Keep the corrected step in the accepted history.
                reasoning_history.append(large_output)
                print_messages.append({"role": "assistant", "content": [{"type": "text", "text": output_text}]})
                print_messages.append({"role": "user", "content": [{"type": "text", "text": f"[REJECTED: Score={utility_score}, Large model corrected]"}]})
            
            # Stop once an answer is present in the accepted history.
            last_step = reasoning_history[-1]
            if 'answer' in last_step:
                break
            
            step_id += 1
            
    except KeyError as e:
        error = handle_exception(e, "KeyError: Missing key -")
        status = 'error'
    except TypeError as e:
        error = handle_exception(e, "TypeError: Invalid type -")
        status = 'error'
    except ValueError as e:
        error = handle_exception(e, "ValueError: Invalid value -")
        status = 'error'
    except json.JSONDecodeError as e:
        error = handle_exception(e, "JSONDecodeError: Invalid JSON -")
        status = 'error'
    except Exception as e:
        error = handle_exception(e, "Unexpected Error:")
        status = 'error'
    
    pred_answer = reasoning_history[-1]
    
    return pred_answer, generated_length, status, error, print_messages


def process_test_type(small_model, small_processor, large_model, large_processor, test_type, data_generator, args):
    if args.benchmark == 'vstar':
        test_path = os.path.join(args.vstar_path, test_type)
        total_count = len([f for f in os.listdir(test_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    elif args.benchmark == 'hr':
        tsv_path = os.path.join(args.hrbench_path, test_type + '.tsv')
        df = pd.read_csv(tsv_path, sep='\t')
        total_count = df.shape[0]
        del df
    elif args.benchmark == 'pope':
        # Load the POPE split once to estimate the sample count.
        file_path = os.path.join(args.pope_path, "Full", f"{test_type}-00000-of-00001.parquet")
        df = pd.read_parquet(file_path)
        total_count = df.shape[0]
        del df
    
    results = []
    start_time = time.time()
    
    # Process the current split sample by sample.
    data_iter = data_generator
    
    with tqdm(total=total_count, desc=f"Processing {test_type}") as pbar:
        while True:
            try:
                data_item = next(data_iter)
            except StopIteration:
                break  # No more samples remain.
            
            # Run the speculative reasoning loop.
            pred_answer, generated_length, status, error, print_messages = speculative_reasoning(
                data_item, small_model, small_processor, large_model, large_processor, args
            )
            
            # Build the stored result payload.
            if args.benchmark == 'vstar':
                result_data = {
                    "image": data_item['img_name'],
                    "question": data_item['question'],
                    "answer": data_item['answer'],
                    "pred_ans": pred_answer,
                }
            elif args.benchmark == 'hr':
                result_data = {
                    "idx": data_item['idx'],
                    "question": data_item['question'],
                    "options": data_item['options'],
                    "answer": data_item['answer'],
                    "answer_str": data_item['answer_str'],
                    "category": data_item['category'],
                    "pred_ans": pred_answer,
                }
            elif args.benchmark == 'pope':
                result_data = {
                    "pid": data_item['pid'],
                    "idx": data_item['idx'],
                    "question_id": data_item['question_id'],
                    "question": data_item['question'],
                    "answer": data_item['answer'],
                    "image_source": data_item['image_source'],
                    "category": data_item['category'],
                    "pred_ans": pred_answer,
                }
            
            results.append({
                "status": status,
                "error": error,
                "generated_length": generated_length,
                "print_messages": print_messages,
                "result": result_data
            })
            
            pbar.update(1)
            safe_cuda_empty_cache()
    
    end_time = time.time()
    if args.verbose:
        print(f"Total time: {end_time - start_time:.2f} seconds")
    
    output_path = os.path.join(args.output_path, f"{args.benchmark}_{test_type}_{args.large_model_path.split('/')[-1]}_{args.small_model_path.split('/')[-1]}_{args.score_threshold}.jsonl")
    
    # Update the per-run latency summary.
    latency_info_path = os.path.join(args.output_path, "latency_summary.json")
    latency_key = output_path.split(args.output_path)[-1].lstrip(os.sep)  # Use a relative path as the key.
    latency_record = {
        latency_key: round(end_time - start_time, 4)
    }

    # Merge with the existing summary when present.
    if os.path.isfile(latency_info_path):
        with open(latency_info_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
        existing.update(latency_record)
        # Keep entries sorted for stable diffs.
        latency_record = dict(sorted(existing.items()))

    with open(latency_info_path, "w", encoding="utf-8") as f:
        json.dump(latency_record, f, ensure_ascii=False, indent=4)

    with open(output_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    if args.verbose:
        print(f"Saved results for {test_type} to {output_path}")
        print(f"Latency saved to {latency_info_path}")


def load_vstar_data_generator(vstar_path, test_type):
    """Load V* samples."""
    test_path = os.path.join(vstar_path, test_type)
    image_files = [f for f in os.listdir(test_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    for img_name in image_files:
        try:
            img_path = os.path.join(test_path, img_name)
            anno_path = os.path.join(test_path, img_name.rsplit('.', 1)[0] + '.json')
            
            if not os.path.exists(anno_path):
                continue
                
            with open(anno_path, 'r') as f:
                anno = json.load(f)
            
            question = anno.get('question', '')
            options = anno.get('options', [])
            question_text = build_mc_question_text(question, options)
            
            data_item = {
                'img_path': img_path,
                'img_name': img_name,
                'question': question,
                'options': options,
                'question_text': question_text,
                'answer': options[0]
            }
            
            yield data_item
            del data_item, anno, question, options, question_text
        except Exception as e:
            print(f"Error processing item {img_name}: {e}")
            continue


def load_hrbench_data_generator(hrbench_path, test_type):
    """Load HR-Bench samples."""
    tsv_path = os.path.join(hrbench_path, test_type + '.tsv')
    df = pd.read_csv(tsv_path, sep='\t')
    tmp_img_root = './eval_code_thyme/_tmp_hr_images'
    
    for idx in range(df.shape[0]):
        try:
            row = df.iloc[idx]
            img_base64 = row['image']
            img_path = decode_base64_to_temp_image(img_base64, os.path.join(tmp_img_root, test_type), idx)
            
            question = row['question']
            options = [row['A'], row['B'], row['C'], row['D']]
            question_text = build_mc_question_text(question, options)
            
            ans_key = row['answer'] if 'answer' in row.index else None
            ans_str = row[ans_key] if ans_key in ['A', 'B', 'C', 'D'] else None
            category = row.get('category', None)
            
            data_item = {
                'idx': idx,
                'img_path': img_path,
                'question': question,
                'options': [f"A. {row['A']}", f"B. {row['B']}", f"C. {row['C']}", f"D. {row['D']}"],
                'question_text': question_text,
                'answer': ans_key,
                'answer_str': ans_str,
                'category': category
            }
            
            yield data_item
            del data_item, row, img_base64, img_path, question, options, question_text, ans_key, ans_str, category
        except Exception as e:
            print(f"Error processing item {idx}: {e}")
            continue


def load_pope_data_generator(pope_path, test_type):
    """
    Load POPE samples.
    """
    import pandas as pd
    from PIL import Image
    import io
    # Build the parquet path for the current split.
    file_path = os.path.join(pope_path, "Full", f"{test_type}-00000-of-00001.parquet")
    
    # Read the parquet file into a dataframe.
    df = pd.read_parquet(file_path)
    rows_len = df.shape[0]
    
    # Persist decoded images to a temporary local folder.
    tmp_img_root = './eval_code_thyme/_tmp_pope_images'
    ensure_dir(os.path.join(tmp_img_root, test_type))
    
    for idx in range(rows_len):
        try:
            row = df.iloc[idx]
            
            # Decode the image payload.
            image_data = row.get('image', None)
            if image_data:
                pil_img = Image.open(io.BytesIO(image_data['bytes']))
                # Save the decoded image to a temporary file.
                img_path = os.path.join(tmp_img_root, test_type, f"{idx:08d}.jpg")
                if pil_img.mode in ('RGBA', 'P'):
                    pil_img = pil_img.convert('RGB')
                pil_img.save(img_path, format='JPEG', quality=95)
            else:
                continue
            
            # Build the yes/no question string.
            question = row.get('question', '')
            question_text = f"Question: {question}\nPlease answer 'yes' or 'no'."
            
            # Build the normalized sample dictionary.
            data_item = {
                'test_type': test_type,
                'idx': idx,
                'pid': row.get('id', ''),
                'question_id': row.get('question_id', ''),
                'question': question,
                'img_path': img_path,
                'answer': row.get('answer', ''),
                'image_source': row.get('image_source', ''),
                'category': row.get('category', ''),
                'options': [],  # POPE is a yes/no benchmark without options.
                'question_text': question_text,
            }
            
            yield data_item
            del data_item, row, pil_img
        except Exception as e:
            print(f"Error processing POPE item {idx}: {e}")
            continue


def load_model_and_processor(args):
    if args.verbose:
        print(f"Loading small model: {args.small_model_path}")
    small_model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.small_model_path,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    small_processor = Qwen3VLProcessor.from_pretrained(args.small_model_path, padding_side="left")
    if args.verbose:
        print("Small model loaded")

    if args.verbose:
        print(f"Loading large model: {args.large_model_path}")
    config = AutoConfig.from_pretrained(args.large_model_path)
    large_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.large_model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",
        config=config
    )
    large_processor = AutoProcessor.from_pretrained(args.large_model_path, padding_side="left")
    if args.verbose:
        print("Large model loaded")

    return small_model, small_processor, large_model, large_processor


def process_benchmark(args, benchmark, test_types, data_generator_func, model_params):
    """
    Process all requested test splits for one benchmark.
    
    Args:
        args: Parsed command-line arguments.
        benchmark: Benchmark name.
        test_types: List of split names.
        data_generator_func: Dataset generator function.
        model_params: Tuple of small and large model resources.
    """
    small_model, small_processor, large_model, large_processor = model_params
    
    for test_type in test_types:
        if args.verbose:
            print(f"\n{'='*50}")
            print(f"Processing test_type: {test_type}")
            print(f"{'='*50}\n")
        
        # Resolve the dataset root for the current benchmark.
        if benchmark == 'vstar':
            data_path = args.vstar_path
        elif benchmark == 'hr':
            data_path = args.hrbench_path
        elif benchmark == 'pope':
            data_path = args.pope_path
        else:
            raise ValueError(f"Unsupported benchmark: {benchmark}")
        
        # Build the generator and process the selected split.
        data_generator = data_generator_func(data_path, test_type)
        process_test_type(small_model, small_processor, large_model, large_processor, test_type, data_generator, args)


def main():
    args = parse_arguments()
    os.makedirs(args.output_path, exist_ok=True)

    small_model, small_processor, large_model, large_processor = load_model_and_processor(args)
    model_params = (small_model, small_processor, large_model, large_processor)
    
    if args.benchmark == 'vstar':
        if args.test_type == 'all':
            process_benchmark(args, 'vstar', ['direct_attributes', 'relative_position'], load_vstar_data_generator, model_params)
        else:
            process_benchmark(args, 'vstar', [args.test_type], load_vstar_data_generator, model_params)
    elif args.benchmark == 'hr':
        if args.test_type == 'all':
            process_benchmark(args, 'hr', ['hr_bench_4k', 'hr_bench_8k'], load_hrbench_data_generator, model_params)
        else:
            process_benchmark(args, 'hr', [args.test_type], load_hrbench_data_generator, model_params)
    elif args.benchmark == 'pope':
        if args.test_type == 'all':
            process_benchmark(args, 'pope', ['adversarial', 'popular', 'random'], load_pope_data_generator, model_params)
        else:
            process_benchmark(args, 'pope', [args.test_type], load_pope_data_generator, model_params)
    else:
        raise ValueError(f"Unsupported benchmark: {args.benchmark}")


if __name__ == '__main__':
    main()
