import generation as g
import pandas as pd
import prompts_construction as pc
import evaluation as ev
from google import genai


n = 770 #max is 771
alias_to_full, answers, prob = ev.Get_prompts_for_LLM(n)

#ID_restore, ans, ex = ev.Get_prompts_for_LLM(n)
gen_client = genai.Client()
fixed_prompt = 'fixed'
gen_file = "LLM_auto_responses"
#flash = 'gemini-2.5-flash'
pro = 'gemini-2.5-pro'

# LLM_answers_file = g.generate(fixed_prompt, ex, n, gen_client, gen_file, flash, delay = 10)
LLM_answers_file = g.generate(fixed_prompt, prob, n, gen_client, gen_file, pro, delay = 10)
LLM_answers = ev.read_json(LLM_answers_file)
#result, strict, loose, perc_entailment = ev.checK_LLM(LLM_answers, ans)
result, strict, loose, perc_entailment, len_metrics, len_scores = ev.checK_LLM(LLM_answers, answers, alias_to_full)

print("-----------------------------------")
print(f"Strict scores: ", strict)
print(f"Loose scores: ", loose)
print(f"Percentage entailment", perc_entailment)
print("Number of pairIDs:", g.count_pairIDs(f'final_{gen_file}.json'))
print("Length metrics: ", len_metrics)
print("Length Scores: ", len_scores)
