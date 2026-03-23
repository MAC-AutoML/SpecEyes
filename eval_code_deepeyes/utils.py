import math
from io import BytesIO
import base64
from PIL import Image
import io
import torch
import gc

IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
tc_start_token = "<tool_call>"
tc_end_token = "</tool_call>"
abc_map = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F'}
stop_strings = ["</tool_call>", "</answer>"]

def round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor

def ceil_by_factor(number: int, factor: int) -> int:
    return math.ceil(number / factor) * factor

def floor_by_factor(number: int, factor: int) -> int:
    return math.floor(number / factor) * factor

def smart_resize(height: int, width: int, factor: int = IMAGE_FACTOR, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS) -> tuple[int, int]:
    """Preserve the original smart resize heuristic."""
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar

def map_box(left, top, right, bottom, width_scale, height_scale, ori_width, ori_height):
    left = int(left * width_scale)
    top = int(top * height_scale)
    right = int(right * width_scale)
    bottom = int(bottom * height_scale)
    left = max(left, 0)
    top = max(top, 0)
    right = min(right, ori_width)
    bottom = min(bottom, ori_height)
    return left, top, right, bottom

def encode_pil_image_to_base64(pil_image):
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str

def decode_base64_to_image(base64_string, target_size=-1):
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    if image.mode in ('RGBA', 'P'):
        image = image.convert('RGB')
    if target_size > 0:
        image.thumbnail((target_size, target_size))
    return image

def safe_cuda_empty_cache():
    try:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()
    except Exception:
        pass

def token_separability(logits_t, top_k=64, eps=1e-6):
    """
    logits_t: Tensor of shape [vocab_size]
    return: scalar separability score
    """
    # Select the top-k logits.
    topk_vals, _ = torch.topk(logits_t, k=top_k)

    # Score of the greedy token.
    z1 = topk_vals[0]

    # Competing tokens excluding the top-1 choice.
    competitors = topk_vals[1:]

    mu = competitors.mean()
    std = competitors.std(unbiased=False)

    score = (z1 - mu) / (std + eps)
    return score

def answer_separability(logits, top_k=64, mode="min", alpha=0.6):
    """
    logits: Tensor of shape [n, vocab_size]
    mode: "min" | "mean" | "bottom20"
    """
    scores = []
    for t in range(logits.size(0)):
        s = token_separability(
            logits[t],
            top_k=top_k
        )
        scores.append(s)

    scores = torch.stack(scores)

    if mode == "min":
        score = scores.min()
        return torch.sigmoid(alpha * score)

    elif mode == "mean":
        score = scores.mean()
        return torch.sigmoid(alpha * score)

    elif mode == "bottom20":
        k = max(1, int(0.2 * len(scores)))
        score = torch.topk(scores, k=k, largest=False).values.mean()
        return torch.sigmoid(alpha * score)
    else:
        raise ValueError(f"Unknown aggregation mode: {mode}")
