# Third-Party Notices

## ms-swift sandbox logic

`eval_code_thyme/sandbox.py` is derived from a localized sandbox implementation used during the original experiments.

- Upstream project: `modelscope/ms-swift`
- Upstream URL: `https://github.com/modelscope/ms-swift`
- Local status: copied and modified for this release so that Thyme evaluation no longer depends on a vendored full `swift/` tree

Please review the upstream project license terms if you further redistribute modified versions of that file.

## qwen-vl-utils

This repository includes a replacement implementation in `scripts/fetch_image.py` that is intended to patch the installed `qwen-vl-utils` package via `scripts/patch_qwen_vl_utils.py`.
