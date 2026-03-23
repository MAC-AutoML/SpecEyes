#!/usr/bin/env python3

import importlib.util
import inspect
import sys
from pathlib import Path


def find_fetch_image_bounds(lines):
    start = None
    end = None
    for idx, line in enumerate(lines):
        if line.startswith("def fetch_image("):
            start = idx
            break
    if start is None:
        raise RuntimeError("Could not find `def fetch_image(` in qwen_vl_utils.vision_process.")

    for idx in range(start + 1, len(lines)):
        line = lines[idx]
        if line.startswith("def ") and not line.startswith("def fetch_image("):
            end = idx
            break
    if end is None:
        end = len(lines)
    return start, end


def main():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

    spec = importlib.util.find_spec("qwen_vl_utils.vision_process")
    if spec is None or spec.origin is None:
        raise RuntimeError("qwen_vl_utils.vision_process is not installed in the current environment.")

    import fetch_image as replacement_module

    target_path = Path(spec.origin)
    replacement_source = inspect.getsource(replacement_module.fetch_image).rstrip() + "\n\n"
    target_lines = target_path.read_text(encoding="utf-8").splitlines(keepends=True)
    start, end = find_fetch_image_bounds(target_lines)
    patched_text = "".join(target_lines[:start]) + replacement_source + "".join(target_lines[end:])
    target_path.write_text(patched_text, encoding="utf-8")
    print(f"Patched fetch_image in {target_path}")


if __name__ == "__main__":
    main()
