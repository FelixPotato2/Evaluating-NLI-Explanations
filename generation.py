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

    with open(generated_file + '.txt', "r", encoding=encoding) as f:
        text = f.read()

    text = re.sub(r"^`{3,}\s*json\s*$", "", text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r"^`{3,}\s*$", "", text, flags=re.MULTILINE)

    decoder = json.JSONDecoder()
    i = 0
    merged = {}

    while True:
        while i < len(text) and text[i].isspace():
            i += 1
        if i >= len(text):
            break

        obj, end = decoder.raw_decode(text, i)
        if not isinstance(obj, dict):
            raise ValueError(f"Expected a JSON object at position {i}, got {type(obj)}")

        merged.update(obj)
        i = end

    with open(f"final_{generated_file}.json", "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    

n = 5 #max is 771
ID_restore, answers, ex = ev.Get_prompts_for_LLM(n)
gen_client = genai.Client()
fixed_prompt = 'fixed'
gen_file = "LLM_auto_responses"
flash = "gemini-2.0-flash"

generate(fixed_prompt, ex, n, gen_client, gen_file, flash)

# with open('fixed.txt', 'r') as file:
#     fixed = file.read()

# prompts = pc.construct_prompt(fixed, examples, amount)

# client = genai.Client()

# open("LLM_auto_responses.txt", "w").close()
# for i, prompt in enumerate(prompts):
#     try:
#         response = client.models.generate_content(
#             model="gemini-2.0-flash",
#             contents=prompt
#         )   
#         with open("LLM_auto_responses.txt", "a", encoding="utf-8") as resp:
#             resp.write(response.text)
#     except Exception as e:
#         print("Error at", i, e)
#         time.sleep(10)  
#         continue

#     time.sleep(1.0)  


# with open("LLM_auto_responses.txt", "r", encoding="utf-8") as f:
#     text = f.read()

# text = re.sub(r"^`{3,}\s*json\s*$", "", text, flags=re.IGNORECASE | re.MULTILINE)
# text = re.sub(r"^`{3,}\s*$", "", text, flags=re.MULTILINE)

# decoder = json.JSONDecoder()
# i = 0
# merged = {}

# while True:

#     while i < len(text) and text[i].isspace():
#         i += 1
#     if i >= len(text):
#         break

#     obj, end = decoder.raw_decode(text, i)  
#     if not isinstance(obj, dict):
#         raise ValueError(f"Expected a JSON object at position {i}, got {type(obj)}")

#     merged.update(obj) 
#     i = end

# with open("LLM_auto.json", "w", encoding="utf-8") as f:
#     json.dump(merged, f, ensure_ascii=False, indent=2)