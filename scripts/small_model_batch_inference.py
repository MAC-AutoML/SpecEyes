import argparse
import os
import json
import copy
import torch
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor, Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from eval_code_deepeyes.utils import *
from eval_code_deepeyes.prompt import *

# Option labels used by V*.
abc_map = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F'}



# V* data loader.
def load_vstar_data_generator(vstar_path, test_type):
    """
    Load V* samples.

    Args:
        vstar_path: Dataset root.
        test_type: Split name such as 'direct_attributes' or 'relative_position'.
    """
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

# HR-Bench data loader.
def load_hrbench_data_generator(hrbench_path, test_type):
    """
    Load HR-Bench samples.

    Args:
        hrbench_path: Dataset root.
        test_type: Split name such as 'hr_bench_4k' or 'hr_bench_8k'.
    """
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

# POPE data loader.
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
        
        # Decode the image payload into a PIL image.
        image_data = row.get('image', None)
        pil_img = None
        if image_data is not None:
            # Convert byte payloads to PIL images.
            if isinstance(image_data['bytes'], bytes):
                try:
                    pil_img = Image.open(io.BytesIO(image_data['bytes']))
                except Exception as e:
                    print(f"Error opening image: {e}")
                    # Fall back to a blank image if decoding fails.
                    pil_img = Image.new('RGB', (100, 100), color='white')
            else:
                # Fall back to a blank image for unexpected payloads.
                pil_img = Image.new('RGB', (100, 100), color='white')
        else:
            # Fall back to a blank image when the sample is missing image bytes.
            pil_img = Image.new('RGB', (100, 100), color='white')
        
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

# Answer-prompt helper.
def prepare_messages_to_answer(messages_to_answer, question_prompt, args):
    """
    Prepare the messages used for direct answering.
    """
    USER_PROMPT = "\nAnswer:"
    messages_to_answer[0]["content"] = SYSTEM_PROMPT_ANSWER
    messages_to_answer[1]["content"][-1]["text"] = question_prompt + USER_PROMPT

    return messages_to_answer

# Batched generation helper.
def get_batch_response(messages_list, model, processor, return_probs=False):
    """
    Generate model responses for a batch of messages.
    """
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
        generated_length = sequences.shape[1] - input_length - 1
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
            # Keep only the valid generation steps for this sample.
            sample_logits = torch.stack([logits[t][i] for t in range(actual_len)], dim=0)
            logits_list.append(sample_logits)
    else:
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
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        generated_ids_trimmed = outputs[:, inputs.input_ids.shape[1]:]
        output_texts = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)
    
    return output_texts, logits_list

# Batched small-model evaluation.
def process_batch_small_model(batch_data, model, processor, args):
    """
    Run the small model on a batch and collect confidence signals.
    """
    messages_list = []
    question_prompts = []
    
    # Build answer prompts for the full batch.
    for item in batch_data:
        messages_to_answer = copy.deepcopy(item['messages'])
        messages_to_answer = prepare_messages_to_answer(messages_to_answer, item['question_prompt'], args)
        messages_list.append(messages_to_answer)
        question_prompts.append(item['question_prompt'])
    

    # Run one batched forward pass.
    output_texts, logits_list = get_batch_response(messages_list, model, processor, return_probs=True)
    
    results = []
    for i, (output_text, logits) in enumerate(zip(output_texts, logits_list)):
        conf_infos = {}
        if logits is not None:
            # Confidence based on mean greedy-token probability.
            max_probs, _ = logits.softmax(dim=-1).max(dim=-1)
            conf_log = torch.exp(torch.mean(torch.log(max_probs))).detach().cpu().item()
            conf_infos['log'] = conf_log

            # Confidence based on answer separability.
            for mode in ['min', 'mean', 'bottom20']:
                confidence_score = answer_separability(logits, mode=mode).cpu().item()
                conf_infos[f'{mode}'] = confidence_score
        
        # Extract the answer span when tags are present.
        if '</answer>' in output_text and '<answer>' in output_text:
            pred_answer = output_text.split('<answer>')[1].split('</answer>')[0].strip()
        else:
            pred_answer = output_text
        
        # Match the result schema expected by the judge scripts.
        data_item = batch_data[i].get('data_item', None)
        if data_item:
            if args.benchmark == 'vstar':
                # vstar benchmark
                result_data = {
                    "image": data_item['image_path'],
                    "question": data_item['question'],
                    "answer": data_item['answer'],
                    "pred_ans": pred_answer,
                }
            elif args.benchmark == 'hr':
                # hr benchmark
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
                    "pid": data_item.get('pid', ''),
                    "idx": data_item.get('idx', 0),
                    "question_id": data_item.get('question_id', ''),
                    "question": data_item.get('question', ''),
                    "answer": data_item.get('answer', ''),
                    "image_source": data_item.get('image_source', ''),
                    "category": data_item.get('category', ''),
                    "pred_ans": pred_answer,
                }
        else:
            result_data = {}
        
        results.append({
            'data_item': data_item,
            'output_text': output_text,
            'pred_answer': pred_answer,
            'conf_infos': conf_infos,
            'question_prompt': question_prompts[i],
            'messages': messages_list[i],
            'result_data': result_data  # Preserve the judge-compatible result format.
        })
    
    return results

# Small-model loader.
def load_small_model(args, verbose=False):
    """
    Load the requested small model and processor.
    """
    if verbose:
        print(f"Loading small model: {args.model_path}")
    
    if "Qwen3" in args.model_path:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model_path,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        processor = Qwen3VLProcessor.from_pretrained(args.model_path, padding_side="left")
    elif "Qwen2.5" in args.model_path:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_path,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        processor = AutoProcessor.from_pretrained(args.model_path, padding_side="left")
    else:
        raise ValueError(f"Unsupported model path: {args.model_path}")
    if verbose:
        print("Small model loaded")
    return model, processor

# Sample-to-message conversion.
def prepare_data_item_for_inference(data_item):
    """
    Convert a raw sample into the chat-message format used for inference.
    """
    # Load the image payload.
    if 'image_pil' in data_item:
        # HR-Bench and POPE already provide a PIL image.
        pil_img = data_item['image_pil']
    elif 'image_path' in data_item:
        # V* stores image paths on disk.
        pil_img = Image.open(data_item['image_path'])
    else:
        raise ValueError("Data item must have either 'image_pil' or 'image_path'")
    
    ori_width, ori_height = pil_img.size
    resize_w, resize_h = smart_resize(ori_width, ori_height)
    pil_img_resized = pil_img.resize((resize_w, resize_h))
    base64_image = encode_pil_image_to_base64(pil_img_resized)
    
    # Build the text prompt.
    if data_item.get('query'):
        prompt = data_item['query']
    else:
        if data_item.get('options'):
            if 'option_str' in data_item:
                # Reuse preformatted options when available.
                prompt = f"Question: {data_item['question']}\nOptions: {data_item['option_str']}"
            else:
                # Otherwise format the options on the fly.
                option_str = "\n"
                for option in data_item['options']:
                    option_str += f"{option}\n"
                prompt = f"Question: {data_item['question']}\nOptions: {option_str}"
        else:
            prompt = f"Question: {data_item['question']}\n"
    
    # Build the chat message list expected by Qwen VL models.
    messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
            {"type": "text", "text": prompt}
        ]},
    ]
    
    return {
        'messages': messages,
        'question_prompt': prompt,
        'data_item': data_item
    }

# Main entry point.
def main():
    """
    Run batched inference for the supported benchmarks.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--benchmark', type=str, choices=['vstar', 'hr', 'pope'], default='vstar')
    parser.add_argument('--verbose', action='store_true', help="Enable verbose output")
    parser.add_argument('--output_dir', type=str, default="eval_results_qwen3vl-2b-Instruct")
    # vstar
    parser.add_argument('--vstar_path', type=str, default="data/vstar")
    # hr-bench
    parser.add_argument('--hrbench_path', type=str, default="data/HR-Bench")
    # pope
    parser.add_argument('--pope_path', type=str, default="data/POPE")
    args = parser.parse_args()
    
    start_time = time.time()
    
    # Load the target small model.
    model, processor = load_small_model(args, verbose=args.verbose)
    
    # Supported test splits per benchmark.
    benchmark_test_types = {
        'vstar': ['direct_attributes', 'relative_position'],
        'hr': ['hr_bench_4k', 'hr_bench_8k'],
        'pope': ['adversarial', 'popular', 'random']
    }
    
    # Resolve the test splits for the selected benchmark.
    test_types = benchmark_test_types.get(args.benchmark, [])
    if not test_types:
        raise ValueError(f"No test types defined for benchmark: {args.benchmark}")
    
    # Process each split in sequence.
    for test_type in test_types:
        if args.verbose:
            print(f"\n{'='*60}")
            print(f"Processing {args.benchmark} {test_type} data with batch size {args.batch_size}")
            print(f"{'='*60}")
        
        # Pick the dataset loader for the selected benchmark.
        if args.benchmark == 'vstar':
            data_generator = load_vstar_data_generator(args.vstar_path, test_type)
        elif args.benchmark == 'hr':
            data_generator = load_hrbench_data_generator(args.hrbench_path, test_type)
        elif args.benchmark == 'pope':
            data_generator = load_pope_data_generator(args.pope_path, test_type)
        else:
            raise ValueError(f"Unsupported benchmark: {args.benchmark}")
        
        # Estimate the number of samples for progress reporting.
        total_samples = 0
        if args.benchmark == 'vstar':
            test_path = os.path.join(args.vstar_path, test_type)
            total_samples = len(list(filter(lambda file: '.json' not in file, os.listdir(test_path))))
        elif args.benchmark == 'hr':
            tsv_path = os.path.join(args.hrbench_path, test_type + '.tsv')
            df = pd.read_csv(tsv_path, sep='\t')
            total_samples = df.shape[0]
        elif args.benchmark == 'pope':
            file_path = os.path.join(args.pope_path, "Full", f"{test_type}-00000-of-00001.parquet")
            df = pd.read_parquet(file_path)
            total_samples = df.shape[0]
            del df
        
        
        # Accumulate outputs for the current split.
        batch_data = []
        test_results = []
        batch_latency = []
        processed_samples = 0
        
        # Stream batches with a progress bar.
        with tqdm(total=total_samples, desc=f"Processing {test_type}", unit="sample") as pbar:
            while True:
                # Assemble one batch of input samples.
                current_batch = []
                for _ in range(args.batch_size):
                    try:
                        data_item = next(data_generator)
                        prepared_item = prepare_data_item_for_inference(data_item)
                        current_batch.append(prepared_item)
                    except StopIteration:
                        break
                
                if not current_batch:
                    break  # No more samples remain.
                
                # Run batched inference.
                t1 = time.time()
                batch_results = process_batch_small_model(current_batch, model, processor, args)
                t2 = time.time()
                batch_latency.append(t2 - t1)
                test_results.extend(batch_results)
                
                safe_cuda_empty_cache()
                # Optionally print per-sample outputs.
                if args.verbose:
                    for i, result in enumerate(batch_results):
                        print(f"\nSample {len(test_results) - len(batch_results) + i + 1}:")
                        print(f"【Question】: {result['data_item']['question']}\n")
                        print(f"【Output Text】: {result['output_text']}\n")
                        print(f"【Predicted Answer】: {result['pred_answer']}\n")
                        sys.stdout.flush()
                
                # Update the progress bar.
                processed_samples += len(current_batch)
                pbar.update(len(current_batch))
        
        # Convert outputs to the stored result format.
        results = []
        for result in test_results:
            data_item = result['data_item']
            output_text = result['output_text']
            conf_infos = result['conf_infos']
            pred_answer = result['pred_answer']
            
            # Wrap each sample in the expected result record.
            v7_result = {
                "status": "success",
                "error": "",
                "small_answer": output_text,
                "confidence_score_infos": conf_infos,
                "result": result['result_data']
            }
            results.append(v7_result)
        
        # Save the split results to disk.
        output_filename = f"{args.benchmark}_{test_type}_{args.model_path.split('/')[-1]}_{args.batch_size}.jsonl"
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, output_filename)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in results:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        if args.verbose:
            print(f"Results saved to: {output_path}")
        
        end_time = time.time()
        print(f"\nTest type {test_type} completed. Processed: {len(test_results)} samples. Average batch latency: {np.mean(batch_latency):.4f} seconds")
        print(f"Total time: {end_time - start_time:.4f} seconds")
    
    # Release cached GPU memory before exit.
    safe_cuda_empty_cache()
    

if __name__ == "__main__":
    main()
