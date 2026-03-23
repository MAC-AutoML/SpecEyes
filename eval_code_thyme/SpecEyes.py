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
    from .utils import run_evaluation, smart_resize, encode_pil_image_to_base64, safe_cuda_empty_cache, answer_separability
    from .prompt import *
except ImportError:
    from utils import run_evaluation, smart_resize, encode_pil_image_to_base64, safe_cuda_empty_cache, answer_separability
    from prompt import *


ABC_MAP = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F', 7: 'G', 8: 'H', 9: 'I', 10: 'J'}


def parse_arguments():
    parser = argparse.ArgumentParser(description='Thyme Evaluation Script')
    parser.add_argument('--large_model_path', type=str, default="Kwai-Keye/Thyme-RL")
    parser.add_argument('--small_model_path', type=str, default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument('--benchmark', type=str, choices=['vstar', 'hr', 'pope'], default='vstar')
    parser.add_argument('--test_type', type=str, default="all")
    parser.add_argument('--output_path', type=str, default='eval_results_thyme/SpecEyes')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument("--score_threshold", type=float, default=0.95, help="Acceptance threshold")
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--mode', type=str, choices=['min', 'mean', 'bottom20', 'log'], default='min')
    parser.add_argument('--batch_size', type=int, default=6, help="Batch size for batch processing")
    parser.add_argument('--ablation_phaseI_Ms', action='store_true')
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


def init_messages_judge_tc(prompt, img_path, args):
    pil_img = Image.open(img_path)
    ori_width, ori_height = pil_img.size
    resize_w, resize_h = smart_resize(ori_width, ori_height)
    pil_img_resized = pil_img.resize((resize_w, resize_h), resample=Image.BICUBIC)
    base64_image = encode_pil_image_to_base64(pil_img_resized)
    messages = [
        {"role": "system", "content": PROMPT_TEMPLATES_JTC['reliability_weighted']['system_prompt']},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
            {"type": "text", "text": prompt+PROMPT_TEMPLATES_JTC['reliability_weighted']['user_prompt']},
        ]},
    ]

    return messages, prompt


def prepare_messages_to_answer(messages_to_answer, question_prompt, args):
    instruction_prompt = "\nAnswer:"
    # Replace the system prompt.
    messages_to_answer[0]["content"] = SYSTEM_PROMPT_ANSWER
    # Append a direct answer instruction.
    messages_to_answer[1]["content"][-1]["text"] = question_prompt + instruction_prompt

    return messages_to_answer


def get_response(messages, model, processor, image_patch_size=14, return_probs=False, short_answer=False):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages, image_patch_size=image_patch_size)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, return_tensors="pt").to(model.device)

    logits = None
    if return_probs:
        generation_config = {
            'max_new_tokens': 20,
            'do_sample': False,
            'return_dict_in_generate': True,
            'output_logits': True
        }
        with torch.no_grad():
            output = model.generate(
                **inputs,
                **generation_config,
            )
        if isinstance(output, tuple):
            outputs = output[0]
        sequences, logits = output.sequences, output.logits

        input_length = inputs.input_ids.shape[1]
        generated_length = sequences.shape[1] - input_length - 1
        generated_ids_trimmed = sequences[:, input_length:]
        output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    else:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512 if not short_answer else 10,
                do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                stop_strings=["</code>", "</answer>", "<|im_end|>"],
                tokenizer=processor.tokenizer,
            )
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        generated_ids_trimmed = outputs[:, inputs.input_ids.shape[1]:]
        output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]


    return output_text, logits


def get_batch_response(messages_list, model, processor, image_patch_size=14, return_probs=False, short_answer=False):
    """Generate batched model responses."""
    texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages_list]
    image_inputs, video_inputs = process_vision_info(messages_list)
    
    # Tokenize the full batch at once for efficient inference.
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        padding=True
    ).to(model.device)
    
    logits_list = None
    if return_probs:
        generation_config = {
            'max_new_tokens': 20,
            'do_sample': False,
            'return_dict_in_generate': True,
            'output_logits': True
        }
        with torch.no_grad():
            output = model.generate(
                **inputs,
                **generation_config,
            )
        if isinstance(output, tuple):
            outputs = output[0]
        sequences, logits = output.sequences, output.logits
        
        input_length = inputs.input_ids.shape[1]
        generated_ids_trimmed = sequences[:, input_length:]
        output_texts = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        
        # Each logits entry has shape [batch_size, vocab_size].
        logits_list = []
        generated_length = len(logits)
        batch_size = len(messages_list)
        for i in range(batch_size):
            input_length = inputs.input_ids.shape[1]
            seq = sequences[i]
            pad_token_id = processor.tokenizer.pad_token_id
            eos_token_id = processor.tokenizer.eos_token_id
            actual_len = generated_length
            for t in range(input_length, len(seq)):
                if seq[t] == pad_token_id or seq[t] == eos_token_id:
                    actual_len = t - input_length
                    break
            # Keep only valid generation steps for this sample.
            sample_logits = torch.stack([logits[t][i] for t in range(actual_len)], dim=0)
            logits_list.append(sample_logits)
    else:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512 if not short_answer else 10,
                do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                stop_strings=["</code>", "</answer>", "<|im_end|>"],
                tokenizer=processor.tokenizer,
            )
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        generated_ids_trimmed = outputs[:, inputs.input_ids.shape[1]:]
        output_texts = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)
    
    return output_texts, logits_list


def process_batch_small_model(batch_data, small_model, small_processor, args):
    """Run the small model on a batch and collect confidence scores."""
    messages_list = []
    question_prompts = []
    
    # Build answer prompts for the full batch.
    for item in batch_data:
        messages_to_answer = copy.deepcopy(item['messages'])
        messages_to_answer = prepare_messages_to_answer(messages_to_answer, item['question_prompt'], args)
        messages_list.append(messages_to_answer)
        question_prompts.append(item['question_prompt'])
    
    # Run one batched forward pass.
    output_texts, logits_list = get_batch_response(messages_list, small_model, small_processor, return_probs=True)
    
    results = []
    for i, (output_text, logits) in enumerate(zip(output_texts, logits_list)):
        if logits is not None:
            if args.mode == 'log':
                # Confidence based on mean greedy-token probability.
                max_probs, _ = logits.softmax(dim=-1).max(dim=-1)
                confidence_score = torch.exp(torch.mean(torch.log(max_probs))).detach().cpu().item()
            else:
                confidence_score = answer_separability(logits, mode=args.mode).cpu().item()
        else:
            confidence_score = 0.0
        
        batch_data[i]['print_messages'].append(
            {"role": "assistant", "content": output_text}
        )

        results.append({
            'data_item': batch_data[i]['data_item'],
            'output_text': output_text,
            'confidence_score': confidence_score,
            'question_prompt': question_prompts[i],
            'messages': messages_list[i],
            'judge_tc': batch_data[i]['judge_tc'],
            'generated_length': logits.shape[0],
        })
    
    return results


def process_single_large_model(item, large_model, large_processor, args):
    """Run one full Thyme evaluation call for a sample."""
    output_text = ""
    status = "success"
    error = ""
    print_conversation_history = []
    
    try:
        data_item = item['data_item']
        question_text = item['question_text']
        img_path = item['img_path']
        
        # Delegate the full reasoning loop to the Thyme evaluator.
        _, final_answer, generated_length, print_conversation_history, status = run_evaluation(question_text, img_path, large_model, large_processor, args.verbose)
        output_text = final_answer
        
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
    
    return output_text, generated_length, status, error, print_conversation_history


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
    yes_cnt = 0
    no_cnt = 0
    start_time = time.time()
    
    # Stream data in batches for the selected split.
    data_iter = data_generator
    
    with tqdm(total=total_count, desc=f"Processing {test_type}") as pbar:
        while True:
            try:
                # Assemble one batch of samples.
                current_batch = []
                batch_messages = []
                batch_print_messages = []
                batch_question_prompts = []
                batch_question_texts = []
                batch_img_paths = []
                
                # Pull raw samples from the generator.
                for _ in range(args.batch_size):
                    try:
                        data_item = next(data_iter)
                        current_batch.append(data_item)
                    except StopIteration:
                        break
                
                if not current_batch:
                    break  # No more samples remain.
                
                # Build Phase-I judge prompts for the whole batch.
                for data_item in current_batch:
                    img_path = data_item['img_path']
                    question_text = data_item['question_text']
                    messages, question_prompt = init_messages_judge_tc(question_text.split('Please')[0], img_path, args)
                    
                    batch_messages.append(messages)
                    batch_print_messages.append([])  # Thyme does not build printable traces here.
                    batch_question_prompts.append(question_prompt)
                    batch_question_texts.append(question_text)
                    batch_img_paths.append(img_path)
                
                # Run the Phase-I router.
                if not args.baseline:
                    if args.ablation_phaseI_Ms:
                        batch_output_texts, _ = get_batch_response(batch_messages, small_model, small_processor, short_answer=True)
                    else:
                        batch_output_texts, _ = get_batch_response(batch_messages, large_model, large_processor, short_answer=True)
                else:
                    batch_output_texts = ["yes"] * len(current_batch)

                safe_cuda_empty_cache()
                # Split samples by the Phase-I decision.
                small_model_batch = []  # Samples routed to the small model.
                large_model_batch = []  # Samples routed to the large model.
                
                for i, (data_item, output_text, messages, print_messages, question_prompt, question_text, img_path) in enumerate(
                    zip(current_batch, batch_output_texts, batch_messages, batch_print_messages, batch_question_prompts, 
                        batch_question_texts, batch_img_paths)):
                    
                    if output_text.lower().startswith("no"):
                        # Route to the small model branch.
                        small_model_batch.append({
                            'data_item': data_item,
                            'messages': messages,
                            'print_messages': print_messages,
                            'question_prompt': question_prompt,
                            'question_text': question_text,
                            'img_path': img_path,
                            'judge_tc': "no"
                        })
                        no_cnt += 1
                    else:
                        # Route to the full large-model branch.
                        large_model_batch.append({
                            'data_item': data_item,
                            'messages': messages,
                            'print_messages': print_messages,
                            'question_prompt': question_prompt,
                            'question_text': question_text,
                            'img_path': img_path,
                            'judge_tc': "yes",
                            'confidence_score': -1
                        })
                        yes_cnt += 1
                # Evaluate all small-model candidates together.
                if small_model_batch:
                    small_model_results = process_batch_small_model(small_model_batch, small_model, small_processor, args)
                    
                    for item, result in zip(small_model_batch, small_model_results):
                        data_item = result['data_item']
                        output_text = result['output_text']
                        confidence_score = result['confidence_score']
                        question_prompt = result['question_prompt']
                        judge_tc = result['judge_tc']
                        
                        if confidence_score > args.score_threshold:
                            # Accept the small-model answer when confidence is high.
                            answer_model = "small"
                            status = "success"
                            error = ""
                            print_conversation_history = []
                            
                            # Extract the answer span when tags are present.
                            if '</answer>' in output_text and '<answer>' in output_text:
                                pred_answer = output_text.split('<answer>')[1].split('</answer>')[0].strip()
                            else:
                                pred_answer = output_text
                            
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
                                "messages": print_conversation_history,
                                "small_answer": output_text,
                                "confidence_score": confidence_score,
                                "use_model": answer_model,
                                "generated_length": result['generated_length'],
                                "judge_tc": judge_tc,
                                "result": result_data
                            })
                        else:
                            item['confidence_score'] = confidence_score
                            # Fall back to the large model when confidence is too low.
                            large_model_batch.append(item)
                
                safe_cuda_empty_cache()
                # Run the full large-model loop for the remaining samples.
                for item in large_model_batch:
                    data_item = item['data_item']
                    # Delegate the full reasoning loop to the Thyme evaluator.
                    output_text, generated_length, status, error, print_conversation_history = process_single_large_model(
                        item, large_model, large_processor, args
                    )
                    
                    answer_model = "large"
                    pred_answer = output_text
                    
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
                        "messages": print_conversation_history,
                        "small_answer": '',
                        "confidence_score": item['confidence_score'],
                        "use_model": answer_model,
                        "generated_length": generated_length,
                        "judge_tc": item['judge_tc'],
                        "result": result_data
                    })
                
                pbar.update(len(current_batch))
                safe_cuda_empty_cache()
            except Exception as e:
                handle_exception(e, "Error processing batch:")
                pbar.update(len(current_batch))
                safe_cuda_empty_cache()

    end_time = time.time()
    if args.verbose:
        print(f"Total time: {end_time - start_time:.2f} seconds")
    if args.baseline:
        args.small_model_path = "baseline"
        args.score_threshold = "None"
        args.mode = "None"
    output_path = os.path.join(args.output_path, f"{args.benchmark}_{test_type}_{args.large_model_path.split('/')[-1]}_{args.small_model_path.split('/')[-1]}_{args.batch_size}_{args.mode}_{args.score_threshold}.jsonl")
    if args.ablation_phaseI_Ms:
        output_path = output_path.replace(".jsonl", f"_PhaseIMs.jsonl")
        
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
    if args.verbose:
        print(f"Saved results for {test_type} to {output_path}")
        print(f"Yes count: {yes_cnt}, No count: {no_cnt}")


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
    if "Qwen3" in args.small_model_path:
        small_model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.small_model_path,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        small_processor = Qwen3VLProcessor.from_pretrained(args.small_model_path, padding_side="left")
    elif "Qwen2.5" in args.small_model_path:
        small_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.small_model_path,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        small_processor = AutoProcessor.from_pretrained(args.small_model_path, padding_side="left")
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
        # ['direct_attributes', 'relative_position']
        if args.test_type == "all":
            process_benchmark(args, 'vstar', ['direct_attributes', 'relative_position'], load_vstar_data_generator, model_params)
        else:
            process_benchmark(args, 'vstar', [args.test_type], load_vstar_data_generator, model_params)
    elif args.benchmark == 'hr':
        if args.test_type == "all":
            process_benchmark(args, 'hr', ['hr_bench_4k', 'hr_bench_8k'], load_hrbench_data_generator, model_params)
        else:
            process_benchmark(args, 'hr', [args.test_type], load_hrbench_data_generator, model_params)
    elif args.benchmark == 'pope':
        # POPE includes three official splits.
        if args.test_type == "all":
            process_benchmark(args, 'pope', ['adversarial', 'popular', 'random'], load_pope_data_generator, model_params)
        else:
            process_benchmark(args, 'pope', [args.test_type], load_pope_data_generator, model_params)
    else:
        raise ValueError(f"Unsupported benchmark: {args.benchmark}")


if __name__ == '__main__':
    main()
