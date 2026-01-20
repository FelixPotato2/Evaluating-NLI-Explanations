import pandas as pd
import re
import string
from explore_esnli_data import print_example

#df = pd.read_csv("entailment_probs_2.csv")
df = pd.read_csv("entailment_probs_or.csv")

# correct answers: Sentence1_Highlighted_Ordered_{i}: is a type of Sentence2_Highlighted_Ordered_{i}:
problem_dict ={}

def get_LLM_problems(df, nr_problems, example = False, seed = 773):
    problems_dict = {}
    if example: 
        ex_df = df.iloc[[0]]
        answer_dict_ex = get_correct_answers(ex_df)
        pair_dict_ex = {
        row["pairID"]: {
        "premise": row["Sentence1"],
        "hypothesis": row["Sentence2"]
        }
        for i, row in ex_df.iterrows()
        }
        df.drop(0)
        #sampled_df = df.iloc[0]
    else:
        answer_dict_ex = None
        pair_dict_ex = None
    
    sampled_df = df.sample(n=nr_problems, random_state=seed) 

    pair_dict = {
        row["pairID"]: {
        "premise": row["Sentence1"],
        "hypothesis": row["Sentence2"]
    }
    for i, row in sampled_df.iterrows()
    }
    answer_dict = get_correct_answers(sampled_df)
    
    return pair_dict, answer_dict, pair_dict_ex, answer_dict_ex

def get_overlap(text, highlighted):
    result = []
    for word in text:
        word = word.strip(" ,.")
        if word.lower() in ["a", "an", "the"]:
            continue
        if word in highlighted:
            result.append(word)
    return result


def get_correct_answers(df):
    answers_dict = {}

    for idx, row in df.iterrows():
        ann_matches = row["matching_explanations"].split(",")

        answers = []
        #loop over the annotators who used keywords
        for ann_i in ann_matches:
            
           
            splitted_ex = re.split(r"(?:type of|form of| kind of)", row[f"Explanation_{ann_i}"])
            
            for j in range(0, len(splitted_ex)-1):
                #This is a bit ugly if there are two (or even more) type of relations as the list element 1 contains both the right part of the first relation and the left part of the second, hard to split. 
                left = splitted_ex[j]
                right = splitted_ex[j + 1]
                # overlap tussen Sentence1_Highlighted_Ordered_{i} en left en Sentence2_Highlighted_Ordered_{i}
                left_overlap = get_overlap(left.split(), row[f"Sentence1_Highlighted_Ordered_{ann_i}"]) 
                right_overlap = get_overlap(right.split(), row[f"Sentence2_Highlighted_Ordered_{ann_i}"])
                if len(left_overlap)>0 and len(right_overlap)>0: 
                    answers.append({
                        "left": left_overlap,
                        "right": right_overlap,
                        "annotator_nr": ann_i
                    })
                #signal with an else here the possibility of no answer, or check below

        answers_dict[row["pairID"]] = answers

    return answers_dict


def check_LLM(answers, LLM_output):
    result = {}
    for pairID in answers:
        #this is not a nice place to check this so improved in newer version
        if len(answers[pairID]) == 0: 
            result[pairID] = "No annotator answer"
        #check in case the LLM does not follow the instructions and misses an answer
        elif pairID not in LLM_output:
            result[pairID] = "Not answered"
        else:
            for LLM_answer in LLM_output[pairID]:
                splitted_ans = LLM_answer.split("is a type of")
                spl_str_ans = [word.strip(" ,.") for word in splitted_ans if word.strip(" ,.") not in ["a", "an", "the"]]
                #exact match:
                exact_match = 0
                partial_match = 0
                for answer in answers[pairID]:
                    if answer["left"] == spl_str_ans[0] and answer["right"] ==spl_str_ans[1]:
                        exact_match +=1 
                    #checks for partial match, but not which one is part of which. Do I want to know??
                    elif bool(set(answer["left"]) & set(spl_str_ans[0])) & bool(set(answer["right"]) & set(spl_str_ans[1])):
                        partial_match +=1
                #now a percentage of how many matches, but I would like to check how many of those answers are actually different and check those actaully different ones
                #TODO actually different answers
                result[pairID] = {"exact": exact_match/len(answers), "partial": partial_match/len(answers), "wrong": len(answers)-(partial_match+exact_match), "total:": len(answers)}
                    
                        
                    

    


problems, answers, problems_ex, answers_ex = get_LLM_problems(df, 5, True)
print(f"problems: {problems}\n")
#print(f"answers{answers}\n")
print(f"problem: {problems_ex} \n answer: {answers_ex}\n")

for pairID in problems.keys():
    print_example(df, ID = pairID)
manual_LLM_answer_ex = {'pairID_0': ["man is a type of person", "black suit is a type of suit"]}
#Todo
"""write function that given a dict with problems and everything generates an LLM prompt with all the problems in that dict """