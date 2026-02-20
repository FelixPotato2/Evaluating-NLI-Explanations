import pandas as pd
from evaluation import get_correct_answers
from explore_esnli_data import print_example

"""
Check for how many problems the answer could not be extracted and subtract this to the total number of available problems
"""
#df = pd.read_csv("entailment_probs_or.csv")
#df = pd.read_csv("entailment_probs_test_or.csv")
df = pd.read_csv("merged_entailment.csv")
answers, missing_answers = get_correct_answers(df)
#print all the relevant information for each problem where the answer template could not be extracted
for missing_id in missing_answers:
    print_example(df, ID = missing_id)
print(f"total problems where answer could not be extracted: {len(missing_answers)}")
print(f"total problems = {df.shape[0]}")
print(f"usable problems = {df.shape[0]-len(missing_answers)}")

#print_example(df, ID = "7579633346.jpg#4r1e")
#print_example(df, ID = "3183195653.jpg#0r1e")
#print(answers["4851047049.jpg#3r1e"])
