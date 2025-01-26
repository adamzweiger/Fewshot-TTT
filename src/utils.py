# utils.py

from vllm import SamplingParams

def post_process_answer(answer, answer_format):
    """
    Post-process answers to comply with the formatting rules.
    """
    answer = answer.strip().lower()
    if not answer:
        return answer
    if answer_format == "Answer with only a sequence of words." or answer_format == "Answer with only a sequence of space-separated parentheses.":
        if '\n' in answer:
            answer = answer.split('\n')[0]
        if '-' in answer:
            answer = answer.split('-')[0]
    else:
        answer = answer.split()[0]

    if '.' in answer:
        answer = answer.split('.')[0].strip()

    if answer_format == "Answer with only the corresponding letter (e.g. (A)).":
        if answer and answer[0] != "(":
            answer = "(" + answer[0] + ")"

    return answer

def compute_accuracy(predictions, targets):
    """
    Compare predictions and targets (case-insensitive match).
    Return accuracy as a percentage (float).
    """
    correct = 0
    for pred, target in zip(predictions, targets):
        if pred.strip().lower() == target.strip().lower():
            correct += 1
    return (correct / len(targets)) * 100 if len(targets) > 0 else 0.0

def inference_vllm(llm,
                   prompts,
                   max_new_tokens=10,
                   task_prompt="",
                   answer_format="",
                   few_shot_prompt_prefix="",
                   lora_request=None):
    """
    Generate predictions with vLLM. Returns a list of strings (predictions).
    """
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_new_tokens,
        stop=["\nQ:"]
    )

    final_prompts = []
    for p in prompts:
        prompt = f"{task_prompt} {answer_format}\n\n{few_shot_prompt_prefix}Q: {p}\nA:"
        final_prompts.append(prompt)

    outputs = llm.generate(final_prompts, sampling_params, lora_request=lora_request)
    predictions = []
    for out in outputs:
        generated = out.outputs[0].text.strip()
        ans = post_process_answer(generated, answer_format)
        predictions.append(ans)
    return predictions
