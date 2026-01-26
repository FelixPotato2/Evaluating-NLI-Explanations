from google import genai
import os
import evaluate_LLMs as ev
import prompts_construction as pc
import time

amount = 10

ID_restore, answers, examples = ev.Get_prompts_for_LLM(amount)

with open('fixed.txt', 'r') as file:
    fixed = file.read()

prompts = pc.construct_prompt(fixed, examples, amount)

client = genai.Client()

open("LLM_auto_responses.txt", "w").close()
for i, prompt in enumerate(prompts):
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        # write response.text somewhere
        with open("LLM_auto_responses.txt", "a", encoding="utf-8") as resp:
            resp.write(response.text)
    except Exception as e:
        print("Error at", i, e)
        time.sleep(10)  
        continue

    time.sleep(0.8)  
#print(response.model_dump_json( exclude_none=True, indent=4))