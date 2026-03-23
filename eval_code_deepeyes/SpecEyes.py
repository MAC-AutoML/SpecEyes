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
    parser.add_argument('--test_type', type=str, default="all")
    parser.add_argument('--output_path', type=str, default='eval_results_deepeyes/SpecEyes')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument("--score_threshold", type=float, default=0.98, help="Acceptance threshold")
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--mode', type=str, choices=['min', 'mean', 'bottom20', 'log'], default='min')
    parser.add_argument('--batch_size', type=int, default=6, help="Batch size for batch processing")
    parser.add_argument('--K', type=int, default=64)
    parser.add_argument('--ablation_K', action='store_true')
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

def init_messages(messages, print_messages, question_prompt):

    instruction_prompt = USER_PROMPT
    # Reset the system prompt.
    messages[0]["content"] = SYSTEM_PROMPT
    print_messages[0]["content"] = SYSTEM_PROMPT
    # Append the answer instruction to the user prompt.
    messages[1]["content"][-1]["text"] = question_prompt + instruction_prompt
    print_messages[1]["content"][-1]["text"] = question_prompt + instruction_prompt

    # Drop any stale assistant output from the printable trace.
    print_messages = print_messages[:2]
    return

def init_messages_judge_tc(data_item, args):
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
        {"role": "system", "content": PROMPT_TEMPLATES_JTC['reliability_weighted']['system_prompt']},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
            {"type": "text", "text": prompt + PROMPT_TEMPLATES_JTC['reliability_weighted']['user_prompt']},
        ]},
    ]

    print_messages = [
        {"role": "system", "content": PROMPT_TEMPLATES_JTC['reliability_weighted']['system_prompt']},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,"}},
            {"type": "text", "text": prompt + PROMPT_TEMPLATES_JTC['reliability_weighted']['user_prompt']},
        ]},
    ]

    return messages, print_messages, prompt, images_pil, wh_infos, base64_image

def prepare_messages_to_answer(messages_to_answer, question_prompt, args):
    instruction_prompt = "\nAnswer:"
    # Replace the system prompt with the answer prompt.
    messages_to_answer[0]["content"] = SYSTEM_PROMPT_ANSWER
    # Append a direct answer instruction.
    messages_to_answer[1]["content"][-1]["text"] = question_prompt + instruction_prompt

    return messages_to_answer

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
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{cropped_pil_image}"}} ,
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

def process_logprobs(response, method, temp=1.0):
    assert len(response.choices[0].logprobs.content) == 1
    token = response.choices[0].logprobs.content[0].token# example: '1', '0'
    token_logprobs = {t.token: t.logprob for t in response.choices[0].logprobs.content[0].top_logprobs}
    # print(f"Original token_logprobs: {token_logprobs}")
    token_logprobs = {k: v for k, v in token_logprobs.items() if k.isdigit()}  # filter out non-digit values

    if method == "greedy":
        # return the vanilla response
        if not token.isdigit():
            return 0
        return int(token)
    elif method == "average":
        # Convert log probabilities to probabilities and normalize each distribution.
        probs = {tok: np.exp(lp / temp) for tok, lp in token_logprobs.items()}
        total_probs = sum(probs.values())
        for tok in probs:
            probs[tok] /= total_probs
        for i in range(10):
            if i not in probs:
                probs[i] = 0
        print(f"Avg score: {sum([int(t) * p for t, p in probs.items()])}")
        return sum([int(t) * p for t, p in probs.items()])
    else:
        raise NotImplementedError

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

def get_batch_response(messages_list, model, processor, image_patch_size=14, return_probs=False, short_answer=False, is_thinking=False):
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
        if is_thinking:
            generation_config['max_new_tokens'] = 1024
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
    if "Thinking" in args.small_model_path:
        is_thinking = True
    else:
        is_thinking = False
    output_texts, logits_list = get_batch_response(messages_list, small_model, small_processor, return_probs=True, is_thinking=is_thinking)
    
    results = []
    for i, (output_text, logits) in enumerate(zip(output_texts, logits_list)):
        if logits is not None:
            if args.mode == 'log':
                # Confidence based on mean greedy-token probability.
                max_probs, _ = logits.softmax(dim=-1).max(dim=-1)
                confidence_score = torch.exp(torch.mean(torch.log(max_probs))).detach().cpu().item()
            else:
                confidence_score = answer_separability(logits, top_k=args.K, mode=args.mode).cpu().item()
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
    """Run the full large-model reasoning loop for one sample."""
    output_text = ""
    status = "success"
    error = ""
    generated_length = 0
    
    try:
        step_id = 0
        # Reset the message state before the full reasoning loop.
        init_messages(item['messages'], item['print_messages'], item['question_prompt'])
        messages = item['messages']
        print_messages = item['print_messages']
        
        while "<answer>" not in output_text:
            if step_id > 5:
                status = "error"
                error = "Step exceeds maximum number of steps"
                raise Exception("Step exceeds maximum number of steps")
            
            output_text, generated_length_step = get_response(messages, large_model, large_processor)
            generated_length += generated_length_step
            
            messages.append({
                "role": "assistant",
                "content": output_text
            })
            print_messages.append({
                "role": "assistant",
                "content": output_text
            })
            
            if "<tool_call>" in output_text:
                messages, print_messages, images_pil, wh_infos = process_messages_to_tc(messages, print_messages, output_text, item['images_pil'], item['wh_infos'])
            else:
                messages.append({
                    "role": "user",
                    "content": USER_PROMPT
                })
                print_messages.append({
                    "role": "user",
                    "content": USER_PROMPT
                })
            
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
    
    item['messages'] = messages
    item['print_messages'] = print_messages

    return output_text, generated_length, status, error

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
                batch_images_pil = []
                batch_wh_infos = []
                batch_base64_images = []
                
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
                    messages, print_messages, question_prompt, images_pil, wh_infos, base64_image = init_messages_judge_tc(data_item, args)
                    
                    batch_messages.append(messages)
                    batch_print_messages.append(print_messages)
                    batch_question_prompts.append(question_prompt)
                    batch_images_pil.append(images_pil)
                    batch_wh_infos.append(wh_infos)
                    batch_base64_images.append(base64_image)
                
                # Run the Phase-I router.
                if args.baseline:   # Baseline mode always routes to the large model.
                    batch_output_texts = ["yes"] * len(current_batch)
                else:
                    if args.ablation_phaseI_Ms:
                        batch_output_texts, _ = get_batch_response(batch_messages, small_model, small_processor, short_answer=True)
                    else:
                        batch_output_texts, _ = get_batch_response(batch_messages, large_model, large_processor, short_answer=True)

                safe_cuda_empty_cache()
                # Split samples by the Phase-I decision.
                small_model_batch = []  # Samples routed to the small model.
                large_model_batch = []  # Samples routed to the large model.
                
                for i, (data_item, output_text, messages, print_messages, question_prompt, images_pil, wh_infos) in enumerate(
                    zip(current_batch, batch_output_texts, batch_messages, batch_print_messages, batch_question_prompts, 
                        batch_images_pil, batch_wh_infos)):
                    
                    if output_text.lower().startswith("no"):
                        # Route to the small model branch.
                        small_model_batch.append({
                            'data_item': data_item,
                            'messages': messages,
                            'print_messages': print_messages,
                            'question_prompt': question_prompt,
                            'images_pil': images_pil,
                            'wh_infos': wh_infos,
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
                            'images_pil': images_pil,
                            'wh_infos': wh_infos,
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
                            
                            # Extract the answer span when tags are present.
                            if '</answer>' in output_text and '<answer>' in output_text:
                                pred_answer = output_text.split('<answer>')[1].split('</answer>')[0].strip()
                            else:
                                pred_answer = output_text
                            
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
                                "small_answer": output_text,
                                "confidence_score": confidence_score,
                                "use_model": answer_model,
                                "generated_length": result['generated_length'],
                                "judge_tc": judge_tc,
                                "print_messages": item['print_messages'],
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
                    # Run the full large-model reasoning loop.
                    output_text, generated_length, status, error = process_single_large_model(
                        item, large_model, large_processor, args
                    )
                    
                    answer_model = "large"
                    
                    if '</answer>' in output_text and '<answer>' in output_text:
                        pred_answer = output_text.split('<answer>')[1].split('</answer>')[0].strip()
                    else:
                        pred_answer = output_text
                    
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
                        "small_answer": '',
                        "confidence_score": item['confidence_score'],
                        "use_model": answer_model,
                        "generated_length": generated_length,
                        "judge_tc": item['judge_tc'],
                        "print_messages": item['print_messages'],
                        "result": result_data
                    })
                
                pbar.update(len(current_batch))
                safe_cuda_empty_cache()
            except Exception as e:
                handle_exception(e, "Error processing batch:")
                safe_cuda_empty_cache()

    end_time = time.time()
    if args.verbose:
        print(f"Total time: {end_time - start_time:.2f} seconds")
    if args.baseline:
        args.small_model_path = "baseline"
        args.score_threshold = "None"
        args.mode = "None"
    output_path = os.path.join(args.output_path, f"{args.benchmark}_{test_type}_{args.large_model_path.split('/')[-1]}_{args.small_model_path.split('/')[-1]}_{args.batch_size}_{args.mode}_{args.score_threshold}.jsonl")
    if args.ablation_K:
        output_path = output_path.replace(".jsonl", f"_K={args.K}.jsonl")
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

def load_models(args):
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
    os.makedirs(args.output_path, exist_ok=True)

    small_model, small_processor, large_model, large_processor = load_models(args)
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



if __name__ == "__main__":
    main()
