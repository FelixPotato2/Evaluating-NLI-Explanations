import generation as g
import prompts_construction as pc
import evaluate_LLMs as ev
from google import genai



n = 100 #max is 771
ID_restore, ans, ex = ev.Get_prompts_for_LLM(n)
gen_client = genai.Client()
fixed_prompt = 'fixed'
gen_file = "LLM_auto_responses"
flash = "gemini-2.0-flash"

LLM_answers_file = g.generate(fixed_prompt, ex, n, gen_client, gen_file, flash, delay = 10)
LLM_answers = ev.read_json(LLM_answers_file)
result, strict, loose, perc_entailment = ev.checK_LLM(LLM_answers, ans)

print("-----------------------------------")
print(f"Strict scores: ", strict)
print(f"Loose scores: ", loose)
print(f"Percentage entailment", perc_entailment)


