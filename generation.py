from google import genai
import os
import evaluate_LLMs as ev
import prompts_construction as pc
import time
import json,re

def generate(fixed_prompt, examples, amount, client, generated_file, model, delay=0.8, encoding="utf-8"):

    with open(fixed_prompt + '.txt', "r", encoding=encoding) as file:
        fixed = file.read()

    prompts = pc.construct_prompt(fixed, examples, amount)
    open(generated_file + '.txt', "w", encoding=encoding).close()

    for i, prompt in enumerate(prompts):
        try:
            response = client.models.generate_content(   # <-- models (plural)
                model=model,
                contents=prompt
            )
            with open(generated_file + '.txt', "a", encoding=encoding) as resp:  # <-- encoding keyword
                resp.write(response.text)
                resp.write("\n")  # helpful separator
        except Exception as e:
            print("Error at", i, e)
            time.sleep(10)
            continue
        time.sleep(delay)

    # Read generated raw output
    with open(generated_file + ".txt", "r", encoding=encoding) as f:
        text = f.read()

    # Extract JSON inside code fences: ```json ... ```
    # Works even if you have 3+ backticks (``` or ``````)
    blocks = re.findall(r"`{3,}\s*json\s*(\{.*?\})\s*`{3,}", text, flags=re.IGNORECASE | re.DOTALL)

    merged = {}
    bad_blocks = 0

    for b in blocks:
        try:
            obj = json.loads(b)
            if not isinstance(obj, dict):
                continue
            merged.update(obj)
        except json.JSONDecodeError:
            bad_blocks += 1
            # skip malformed/truncated blocks

    if not merged:
        raise ValueError("No valid JSON blocks were found in the model output.")

    with open(f"final_{generated_file}.json", "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"Parsed {len(blocks)} fenced blocks; skipped {bad_blocks} malformed blocks.")
    return f"final_{generated_file}.json"