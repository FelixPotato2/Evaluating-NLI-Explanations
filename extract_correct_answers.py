import pandas as pd
import re
import string

df = pd.read_csv("entailment_probs_2.csv")

# correct answers: Sentence1_Highlighted_Ordered_{i}: is a type of Sentence2_Highlighted_Ordered_{i}:
problem_dict ={}

def get_LLM_problems(df, nr_problems, example = False):
    problems_dict = {}
    if example: 
        sampled_df = df.iloc[[0]]
        #sampled_df = df.iloc[0]
    else:
        seed = 773
        sampled_df = df.sample(n=nr_problems, random_state=seed) #TODO make sure example is excluded 
    pair_dict = {
    f"pairID_{i}": {
        "premise": row["Sentence1"],
        "hypothesis": row["Sentence2"]
    }
    for i, row in sampled_df.iterrows()
    }
    answer_dict = get_correct_answers(sampled_df)
    return pair_dict, answer_dict
    # for row in sampled_df:
    #     sampled_df["pairID:"] = {"premise": sampled_df["Sentence1"], "hypothesis": sampled_df["Sentence2"]}


def get_correct_answers(df):
    answers_dict = {}

    for idx, row in df.iterrows():
        matches = row["matching_explanations"].split(",")

        answers = []
        for i in matches:
            i = i.strip()
            answers.append({
                "left": row[f"Sentence1_Highlighted_Ordered_{i}"],
                "right": row[f"Sentence2_Highlighted_Ordered_{i}"],
                "annotator_nr": i
            })

        answers_dict[f"pairID_{idx}"] = answers

    return answers_dict


def check_LLM(df, LLM_output):
    answers = get_correct_answers(df)

    #check if lefthand side of LLM answer is in lefthand side of any? or all? of annotators highlights, same for right side
problems, answers = get_LLM_problems(df, 5)
print(f"problems: {problems}\n")
print(f"answers{answers}\n")

problems_ex, answers_ex = get_LLM_problems(df, 5, True)
print(f"problem: {problems_ex} \n answer: {answers_ex}\n")

manual_LLM_answer_ex = {'pairID_0': ["man is a type of person", "black suit is a type of suit"]}
#Todo
"""write function that given a dict with problems and everything generates an LLM prompt with all the problems in that dict """