SYSTEM_PROMPT_ANSWER = "You are a helpful assistant. Output only the final answer itself. Do not include explanations, reasoning, labels, or any extra text."
SYSTEM_PROMPT_ANSWER_MATH = """
    Solve the math problem.

    Output only:
    <answer>final_answer</answer>

    Read numbers carefully. Use the simplest exact form.
    Extra text = wrong.
"""

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
