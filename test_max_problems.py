import pandas as pd
from evaluate_LLMs import get_correct_answers


#df = pd.read_csv("entailment_probs_or.csv")
df = pd.read_csv("entailment_probs_test_or.csv")
answers, missing_answers = get_correct_answers(df)
print(missing_answers)
print(len(missing_answers))
print(f"total problems = {df.shape[0]}")
print(f"usable problems = {df.shape[0]-len(missing_answers)}")

