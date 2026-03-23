SYSTEM_PROMPT = """You are a helpful assistant.

# Tools
You may call one or more functions to assist with the user query.
You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type":"function","function":{
    "name":"image_zoom_in_tool",
    "description":"Zoom in on a specific region of the input image by cropping it based on a bounding box (bbox) and an optional object label.",
    "parameters":{
        "type":"object",
        "properties":{
            "bbox_2d":{"type":"array","items":{"type":"number"},"minItems":4,"maxItems":4,"description":"The bounding box to zoom in, format: [x1, y1, x2, y2]."},
            "label":{"type":"string","description":"Label of the object (optional)."}
        },
        "required":["bbox_2d"]
    }
}}
</tools>

# How to call a tool
Return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

**Example**:  
<tool_call>  
{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [10, 20, 100, 200], "label": "the apple on the desk"}}  
</tool_call>
"""


SYSTEM_PROMPT_ANSWER = "You are a helpful assistant. Output only the final answer itself. Do not include explanations, reasoning, labels, or any extra text."
SYSTEM_PROMPT_ANSWER_MATH = """
    Solve the math problem.

    Output only:
    <answer>final_answer</answer>

    Read numbers carefully. Use the simplest exact form.
    Extra text = wrong.
"""
SYSTEM_PROMPT_ANSWER_THINKING = """
    You are a helpful assistant.
    
    Think internally but do not reveal your reasoning.

    Your entire response must be inside the <answer> tags.

    Output only:
    <answer>your answer</answer>
"""

USER_PROMPT = "\nThink first, call **image_zoom_in_tool** if needed, then answer. Format strictly as:  <think>...</think>  <tool_call>...</tool_call> (if tools needed)  <answer>...</answer> "

PROMPT_TEMPLATES_JTC = {
    "reliability_weighted": {
        "system_prompt": """You decide based on answer reliability.

Answer "yes" if using a tool would significantly improve correctness or confidence of the answer.

Answer "no" if tools would add little or no value.

Output only "yes" or "no".
""",
        "user_prompt": "\nWould tools significantly improve answer reliability? yes or no."
    },
}
