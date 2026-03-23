import argparse
import os
import json
import pandas as pd
from tqdm import tqdm
import time
import copy
import numpy as np
import traceback
import torch
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor, Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
try:
    from .utils import *
    from .prompt import *
except ImportError:
    from utils import *
    from prompt import *


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--large_model_path', type=str, default="ChenShawn/DeepEyes-7B")
    parser.add_argument('--small_model_path', type=str, default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument('--benchmark', type=str, choices=['vstar', 'hr', 'pope'], default='vstar')
    parser.add_argument('--test_type', type=str, default='all')
    parser.add_argument('--output_path', type=str, default='eval_results_deepeyes/SpecReason')
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

def load_vstar_data_generator(vstar_path, test_type):
    test_path = os.path.join(vstar_path, test_type)
    image_files = list(filter(lambda file: '.json' not in file, os.listdir(test_path)))
    for img in image_files:
        img_path = os.path.join(test_path, img)
        anno_path = os.path.join(test_path, img.replace('.jpg', '.json'))
        with open(anno_path, 'r') as f:
            anno = json.load(f)
        question = anno['question']
        options = anno['options']
        option_str = "\n"
        for i in range(len(options)):
            option_str += abc_map[i + 1] + '. ' + options[i] + '\n'
        data_item = {
            'test_type': test_type,
            'image_path': img_path,
            'image_name': img,
            'question': question,
            'options': options,
            'option_str': option_str,
            'answer': anno['options'][0]
        }
        yield data_item
        del data_item, anno, question, options, option_str

def load_hrbench_data_generator(hrbench_path, test_type):
    tsv_path = os.path.join(hrbench_path, test_type + '.tsv')
    df = pd.read_csv(tsv_path, sep='\t')
    rows_len = df.shape[0]
    for idx in range(rows_len):
        anno = df.iloc[idx]
        img_base64 = anno['image']
        img_pil = decode_base64_to_image(img_base64)
        question = anno['question']
        answer = anno['answer']
        answer_str = anno[answer]
        options = ['A. ' + anno['A'], 'B. ' + anno['B'], 'C. ' + anno['C'], 'D. ' + anno['D']]
        category = anno['category']
        option_str = "\n"
        for i in range(len(options)):
            option_str += options[i] + '\n'
        data_item = {
            'test_type': test_type,
            'idx': idx,
            'image_pil': img_pil,
            'question': question,
            'options': options,
            'option_str': option_str,
            'answer': answer,
            'answer_str': answer_str,
            'category': category
        }
        yield data_item
        del data_item, anno, img_base64, img_pil, question, answer, answer_str, options, category, option_str

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
    
    for idx in range(rows_len):
        row = df.iloc[idx]
        
        # Decode the image payload.
        image_data = row.get('image', None)
        pil_img = Image.open(io.BytesIO(image_data['bytes']))
        
        # Build the normalized sample dictionary.
        data_item = {
            'test_type': test_type,
            'idx': idx,
            'pid': row.get('id', ''),
            'question_id': row.get('question_id', ''),
            'question': row.get('question', ''),
            'answer': row.get('answer', ''),
            'image_source': row.get('image_source', ''),
            'category': row.get('category', ''),
            'image_pil': pil_img,  # Store the decoded PIL image.
            'options': []  # POPE is a yes/no benchmark without options.
        }
        
        yield data_item
        del data_item, row

def init_messages(data_item, args):
    # Prompt template for multiple-choice questions.
    instruction_prompt_with_options = """Question: {question}
        Options: {options}
        """
    # Prompt template for free-form questions.
    instruction_prompt_no_options = """Question: {question}
        """
    
    if args.benchmark == 'vstar':
        # V* always provides answer options.
        option_str = "\n"
        for i in range(len(data_item['options'])):
            option_str += abc_map[i + 1] + '. ' + data_item['options'][i] + '\n'
        prompt = instruction_prompt_with_options.format(question=data_item['question'], options=option_str)

        # Load the image from disk.
        pil_img = Image.open(data_item['image_path'])
    elif args.benchmark == 'hr':
        # HR-Bench always provides answer options.
        option_str = "\n"
        for i in range(len(data_item['options'])):
            option_str += data_item['options'][i] + '\n'
        prompt = instruction_prompt_with_options.format(question=data_item['question'], options=option_str)

        pil_img = data_item['image_pil']
    elif args.benchmark == 'pope':
        # POPE uses a yes/no prompt without options.
        prompt = instruction_prompt_no_options.format(question=data_item['question'])
        
        # Reuse the decoded image produced by the data loader.
        pil_img = data_item['image_pil']
    else:
        raise ValueError(f"Unsupported benchmark: {args.benchmark}")
    
    images_pil = []
    wh_infos = []
    images_pil.append(copy.deepcopy(pil_img))
    ori_width, ori_height = pil_img.size
    resize_w, resize_h = smart_resize(ori_width, ori_height, factor=IMAGE_FACTOR)
    pil_img_resized = pil_img.resize((resize_w, resize_h), resample=Image.BICUBIC)
    width_scale = ori_width / resize_w
    height_scale = ori_height / resize_h
    base64_image = encode_pil_image_to_base64(pil_img_resized)
    wh_infos.append([width_scale, height_scale, ori_width, ori_height])

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
            {"type": "text", "text": prompt + USER_PROMPT},
        ]},
    ]

    print_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,"}},
            {"type": "text", "text": prompt + USER_PROMPT},
        ]},
    ]

    return messages, print_messages, prompt, images_pil, wh_infos, base64_image

def process_messages_to_tc(messages, print_messages, response_message, images_pil, wh_infos):
    action_list = response_message.split("<tool_call>")[1].split("</tool_call>")[0].strip()
    # action_list = eval(action_list)
    try:
        # Parse the tool call payload as a single JSON object.
        action_list = json.loads(action_list)
    except json.JSONDecodeError as e:
        action_list = {}
        messages.append({
            "role": "assistant",
            "content": USER_PROMPT
        })
        print_messages.append({
            "role": "assistant",
            "content": USER_PROMPT
        })
        return messages, print_messages, images_pil, wh_infos


    img_idx = 0  # This pipeline currently assumes a single source image.
    bbox = action_list['arguments']['bbox_2d']
    left, top, right, bottom = bbox
    left, top, right, bottom = map_box(left, top, right, bottom, wh_infos[img_idx][0], wh_infos[img_idx][1], wh_infos[img_idx][2], wh_infos[img_idx][3])
    cropped_image = images_pil[img_idx].crop((left, top, right, bottom))
    images_pil.append(copy.deepcopy(cropped_image))
    ori_width, ori_height = cropped_image.size
    new_w, new_h = smart_resize((right - left), (bottom - top), factor=IMAGE_FACTOR)
    width_scale = ori_width / new_w
    height_scale = ori_height / new_h
    wh_infos.append([width_scale, height_scale, ori_width, ori_height])
    cropped_image = cropped_image.resize((new_w, new_h), resample=Image.BICUBIC)
    cropped_pil_image = encode_pil_image_to_base64(cropped_image)

    next_content = [{"type": "text", "text": "<tool_response>"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{cropped_pil_image}"} } ,
                    {"type": "text", "text": USER_PROMPT},
                    {"type": "text", "text": "</tool_response>"}]
    next_message = [{"role": "user", "content": next_content}]
    messages.extend(next_message)
    
    p_message = [{"role": "user", "content": [
                        {"type": "text", "text": "<tool_response>"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,"}},
                        {"type": "text", "text": USER_PROMPT},
                        {"type": "text", "text": "</tool_response>"}
                    ]}]
    print_messages.extend(p_message)

    return messages, print_messages, images_pil, wh_infos

def get_response(messages, model, processor):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=None,
            stop_strings=["</tool_call>", "</answer>", "<|im_end|>"],
            tokenizer=processor.tokenizer,
        )
    generated_ids_trimmed = outputs[:, inputs.input_ids.shape[1]:]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]
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
    verify_messages.append({"role": "user", "content": verify_prompt})
    
    # Generate the utility score with the large model.
    text = large_processor.apply_chat_template(verify_messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(verify_messages)
    inputs = large_processor(text=[text], images=image_inputs, videos=video_inputs, return_tensors="pt").to(large_model.device)
    
    with torch.no_grad():
        outputs = large_model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=large_processor.tokenizer.pad_token_id,
            eos_token_id=None,
            tokenizer=large_processor.tokenizer,
        )
    
    generated_ids_trimmed = outputs[:, inputs.input_ids.shape[1]:]
    score_text = large_processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]
    
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
    messages, print_messages, question_prompt, images_pil, wh_infos, base64_image = init_messages(data_item, args)
    
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
                speculation_messages.append({"role": "assistant", "content": history_text})
            
            # Let the small model propose the next reasoning step.
            small_output, small_length = get_response(speculation_messages, small_model, small_processor)
            generated_length += small_length
            
            # Step 2: Verify the proposed step with the large model.
            # Build the verification prompt from the same history.
            verification_messages = copy.deepcopy(messages)
            if reasoning_history:
                history_text = "\n".join(reasoning_history)
                verification_messages.append({"role": "assistant", "content": history_text})
            verification_messages.append({"role": "assistant", "content": small_output})
            
            # Ask the large model for a utility score.
            utility_score = get_utility_score(verification_messages, large_model, large_processor)
            
            # Step 3: Accept or replace the proposed step.
            if utility_score >= args.score_threshold:
                # Accept the small-model step.
                reasoning_history.append(small_output)
                print_messages.append({"role": "assistant", "content": small_output})
                print_messages.append({"role": "user", "content": f"[VERIFIED: Score={utility_score}]"})
                
                # Tool-use handling can be added back here if needed.
                # if "<tool_call>" in small_output:
                #     # Execute the tool call and update the message state.
                #     messages, print_messages, images_pil, wh_infos = process_messages_to_tc(
                #         messages, print_messages, small_output, images_pil, wh_infos
                #     )
            else:
                # Reject the step and let the large model replace it.
                # The large model still keeps the ability to issue tool calls.
                correction_messages = copy.deepcopy(messages)
                if reasoning_history:
                    history_text = "\n".join(reasoning_history)
                    correction_messages.append({"role": "assistant", "content": history_text})
                
                # Run one corrective large-model step.
                large_output = ""
                large_length = 0
                current_messages = copy.deepcopy(correction_messages)
                current_print_messages = copy.deepcopy(print_messages)
                current_images_pil = copy.deepcopy(images_pil)
                current_wh_infos = copy.deepcopy(wh_infos)
                
                # Generate a single replacement step.
                output_text, length = get_response(current_messages, large_model, large_processor)
                large_output += output_text
                large_length += length
                
                # Append the replacement step to the working history.
                current_messages.append({"role": "assistant", "content": output_text})
                current_print_messages.append({"role": "assistant", "content": output_text})
                
                # Apply tool calls when the large model requests them.
                if "<tool_call>" in output_text:
                    current_messages, current_print_messages, current_images_pil, current_wh_infos = process_messages_to_tc(
                        current_messages, current_print_messages, output_text, current_images_pil, current_wh_infos
                    )
                else:
                    current_messages.append({"role": "user", "content": USER_PROMPT})
                    current_print_messages.append({"role": "user", "content": USER_PROMPT})
                
                generated_length += large_length
                
                # Keep the corrected step in the accepted history.
                reasoning_history.append(large_output)
                print_messages = current_print_messages
                print_messages.append({"role": "user", "content": f"[REJECTED: Score={utility_score}, Large model corrected]"})
            
            # Stop once an answer tag is produced.
            last_step = reasoning_history[-1]
            if '</answer>' in last_step and '<answer>' in last_step:
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
    
    # Extract the final answer from the accepted history.
    pred_answer = ""
    for step in reasoning_history:
        if '</answer>' in step and '<answer>' in step:
            pred_answer = step.split('<answer>')[1].split('</answer>')[0].strip()
            break
    if not pred_answer and reasoning_history:
        pred_answer = reasoning_history[-1]
    
    return pred_answer, generated_length, status, error, print_messages

def process_test_type(small_model, small_processor, large_model, large_processor, test_type, data_generator, args):
    if args.benchmark == 'vstar':
        test_path = os.path.join(args.vstar_path, test_type)
        total_count = len(list(filter(lambda file: '.json' not in file, os.listdir(test_path))))
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
                    "image": data_item['image_path'],
                    "question": data_item['question'],
                    "answer": data_item['answer'],
                    "pred_ans": pred_answer,
                }
            elif args.benchmark == 'hr':
                result_data = {
                    "idx": data_item['idx'],
                    "question": data_item['question'],
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


def load_models(args):
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
    large_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.large_model_path,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
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
    
    # Ensure the output directory exists.
    os.makedirs(args.output_path, exist_ok=True)

    # Load the small and large models once.
    small_model, small_processor, large_model, large_processor = load_models(args)
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

if __name__ == "__main__":
    main()
