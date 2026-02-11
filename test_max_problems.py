import pandas as pd
#from evaluate_LLMs import get_correct_answers
from evaluation import get_correct_answers
from explore_esnli_data import print_example

#df = pd.read_csv("entailment_probs_or.csv")
#df = pd.read_csv("entailment_probs_test_or.csv")
df = pd.read_csv("merged_entailment.csv")
answers, missing_answers = get_correct_answers(df)
#print(missing_answers)
for missing_id in missing_answers:
    print_example(df, ID = missing_id)
print(len(missing_answers))
print(f"total problems = {df.shape[0]}")
print(f"usable problems = {df.shape[0]-len(missing_answers)}")

#print_example(df, ID = "7579633346.jpg#4r1e")
print_example(df, ID = "1229756013.jpg#0r1e")
print(answers["1229756013.jpg#0r1e"])
