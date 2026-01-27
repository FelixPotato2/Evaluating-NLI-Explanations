import pandas as pd
import re
import string
from explore_esnli_data import print_example
import json
import ast
import math


#df = pd.read_csv("entailment_probs_2.csv")
#df = pd.read_csv("entailment_probs_or.csv")
df = pd.read_csv("merged_entailment.csv")

def get_LLM_problems(df, nr_problems, excluded_ids = set(), example=False, seed = 773):
    """
    Randomly sample problems from a DataFrame.

    Parameters:
    df : pandas.DataFrame
        Input dataframe containing full ESNLI information.
    nr_problems : int
        Number of problems to sample.
    excluded_ids: set
        set of IDs that have already been used and thus should be excluded 
        (specified when recursively calling the function to replace problems with missing answers)
    example : bool
        If True, also extract the first row of df as an example
        for use in an LLM prompt.
    seed: int

    Returns:
    pair_dict : dict
        Dictionary mapping pairIDs to premise-hypothesis pairs.
    answer_dict : dict
        Dictionary mapping pairIDs to correct answers.
    pair_dict_ex : dict or None
        Example problem dictionary (if example=True).
    answer_dict_ex : dict or None
        Example answer dictionary (if example=True).
    """
    
    # example extraction
    if example:
        ex_df = df.iloc[[0]]
        answer_dict_ex, _ = get_correct_answers(ex_df)
        pair_dict_ex = {
            row["pairID"]: {
                "premise": row["Sentence1"],
                "hypothesis": row["Sentence2"],
            }
            for i, row in ex_df.iterrows()
        }
        #remove the first row so that the example won't occur in the problems
        df = df.drop(0)
    else:
        pair_dict_ex = None
        answer_dict_ex = None

    #make sure that previously sampled problems are not sampled again in recursive call
    available_df = df[~df["pairID"].isin(excluded_ids)]

    # Sample problems
    sampled_df = available_df.sample(n=nr_problems, random_state=seed)

    pair_dict = {
        row["pairID"]: {
            "premise": row["Sentence1"],
            "hypothesis": row["Sentence2"],
        }
        for i, row in sampled_df.iterrows()
    }
   
    excluded_ids = excluded_ids.copy()
    excluded_ids.update(pair_dict.keys())

    answer_dict, missing_answers = get_correct_answers(sampled_df)
    #handle missing answers
    if missing_answers:
        #remove the problems that cannot be answered from the sampled dataset
        for pid in missing_answers:
            pair_dict.pop(pid, None)
            answer_dict.pop(pid, None)
        sampled_df = sampled_df[~sampled_df["pairID"].isin(missing_answers)]
        #sample new problems and add them 
        pair_dict2, answer_dict2, _, _ =get_LLM_problems(df,  len(missing_answers), excluded_ids, example= False, seed = seed+1)
        pair_dict.update(pair_dict2)
        answer_dict.update(answer_dict2)

    return pair_dict, answer_dict, pair_dict_ex, answer_dict_ex

def get_correct_answers(df):
    """
    extracts from the human annotators the left and right parts of their mentioned "type of" relation
    If multiple annotators mention the same "type of" relation, but with slightly different wordings
    these explanations are grouped, but both wordings are saved. 

    Parameters:
    df : pandas.DataFrame
        Input dataframe with sampled problems.

    Returns:
    answers_dict : dict
        Dictionary with for each pairID a list of answer groups
        Each answer group in the list is a dictionairy with structure: {"left": [], "right":[], "annotator_nr":[]} 
        the lists within these dictionaries can contain one or multiple phrasings of the same "type of" relation
    missing_answers: set
        Set of IDs of problems that did not have elements in right or left part of answer 
    """
    answers_dict = {}
    missing_answers =set()

    for idx, row in df.iterrows():
        ann_matches = row["matching_explanations"].split(",")
        answers = []
        pair_has_answer = False
        # Loop over annotators who used keywords
        for ann_i in ann_matches:
            splitted_ex = re.split(
                r"(?:type of|form of| kind of)",
                row[f"Explanation_{ann_i}"],
            )

            for j in range(len(splitted_ex) - 1):
                # Note: this way splitting is imperfect when multiple "type of" relations occur,
                # since one segment may contain both the right part of one relation
                # and the left part of the next.
                #max length of 4 to avoid getting any unwanted highlights, 
                # but not the best measure as this might cut off some longer explanations
                left = splitted_ex[j].split()[-6:] 
                right = splitted_ex[j + 1].split()[:4] 

                left_overlap = get_overlap(
                    left,
                    row[f"Sentence1_Highlighted_Ordered_{ann_i}"],
                )
                right_overlap = get_overlap(
                    right,
                    row[f"Sentence2_Highlighted_Ordered_{ann_i}"],
                )

                if left_overlap and right_overlap:
                    pair_has_answer = True
                    grouped = False
                    #group answer with previous answers if they overlap
                    #the grouping makes it even worse if there are words in the left or right part of answer that shouldnt be there
                    for answer in answers:
                        if bool(set(w for group in answer["left"] for w in group) & set(left_overlap)) or bool(set(w for group in answer["right"] for w in group) & set(right_overlap)):
                            answer["left"].append(left_overlap)
                            answer["right"].append(right_overlap)
                            answer["annotator_nr"].append(ann_i)
                            grouped = True
                    if not grouped:   
                        answers.append(
                            {
                                "left": [left_overlap],
                                "right": [right_overlap],
                                "annotator_nr": [ann_i],
                            }
                        )
        #These have to be passed so that new problems can be sampled instead
        if not pair_has_answer:
            missing_answers.add(row["pairID"])
        else:
            answers_dict[row["pairID"]] = answers #??
        #if len(answers[pairID]) > 0:
            #answers_dict[row["pairID"]] = answers

    return answers_dict, missing_answers


def get_overlap(text, highlighted):
    """
    Gets from the text the parts that were highlighted
    
    Parameters: 
    text: list
        list of left or right side of explanation
    highlighted: string
        string of a list of highlighted words from ESNLI explanation
    
    Returns:
    result: list
        list of the words in the text that were also highlighted (excluding articles)
    """
    result = []
    
    highlighted_lower = [h.lower() for h in ast.literal_eval(highlighted) ]

    for word in text:
        cleaned_word = word.strip(" ,.").lower()

        if cleaned_word in ["a", "an", "the"]:
            continue
        for h_string in highlighted_lower:
            #if cleaned_word in h_string:
            if cleaned_word in h_string.split():
                result.append(cleaned_word)
    
    #if result > 1:
        #possible solution for incorrect highlight matches: check if they are next to eachother, otherwise delete the most far one 
    
    return result 


def check_LLM_answer(answers, LLM_output):
    """
    Compares the LLM answers with answers extracted from human annotators and gives performance measures
    Parameters:
    answers: dict
        Dictionary with for each pairID a list of answer groups
        Each answer group in the list is a dictionairy with structure: {"left": [], "right":[], "annotator_nr":[]} 
        the lists within these dictionaries can contain one or multiple phrasings of the same "type of" relation
    LLM_output: dict
        structured as follows:  
        {'3021028400.jpg#1r1e': { ‘answer’: ‘entailment’, ‘explanation’: [‘man is a type of person’, ‘black suit is a type of suit’]}
    

    Returns:
    result: dict
        dictionary with keys pair ids and values dictionaries containing the correctness counts for that pairID
    scores_strict: dict
        dictionary with keys "precision" "recall" and "f1" containing those scores
        considering only the answers that exactly match one of the phrasings of the annotators 

    scores_loose: dict
        dictionary with keys "precision" "recall" and "f1" containing those scores
        considering all answers where there is some overlap in both the right and left side between the answer of the LLM and one of the annotators

    """

    result ={}
    for pairID in answers:
        if pairID not in LLM_output: 
            #result[pairID] = "Not answered"
            # count as unanswered but still include totals so scoring can work
            result[pairID] = {
                "exact": 0,
                "partial": 0,
                "combined_correct": 0,
                "total_answers": len(answers[pairID]),
                "total_LLM_answers": 0
            }
            continue

        else: 
            exact_count = 0
            partial_count = 0

            #Added LLM explanations since the LLM is read into a dict 
            llm_explanations = LLM_output[pairID].get("explanation", [])

            #for LLM_answer in LLM_output[pairID]:
            for LLM_answer in llm_explanations:
                splitted_ans = LLM_answer.split("is a type of")
                if len(splitted_ans) < 2:
                    continue
                spl_str_ans = [word.strip(" ,.") for word in splitted_ans if word.strip(" ,.") not in ["a", "an", "the"]]
                for answer_group_dict in answers[pairID]:
                    left_exact = False
                    right_exact = False
                    left_partial = False
                    right_partial = False
                    
                    for left_answer in answer_group_dict["left"]: 
                        if left_answer == spl_str_ans[0].split():
                            left_exact = True
                        if bool(set(left_answer) & set(spl_str_ans[0].split())):
                            left_partial = True
                    for right_answer in answer_group_dict["right"]:
                        if right_answer == spl_str_ans[1].split():
                            right_exact = True
                        if bool(set(right_answer) & set(spl_str_ans[1].split())):
                            right_partial = True

                    if left_exact and right_exact:
                        exact_count+= 1
                        break #stop iterating over answer group because an exact match is found
                    elif left_partial and right_partial:
                        partial_count +=1
                    
            result[pairID] = { 
                "exact": exact_count,
                "partial": partial_count,
                "combined_correct": exact_count +partial_count,
                "total_answers": len(answers[pairID]),
                "total_LLM_answers": len(LLM_output[pairID])
            }
    scores_strict= calculate_scores(result, "exact")
    scores_loose = calculate_scores(result, "combined_correct")
    return result, scores_strict, scores_loose
                   
def calculate_scores(result, key):
    TP = sum(v[key] for v in result.values() if isinstance(v, dict))
    FP = sum(v["total_LLM_answers"] - v[key] for v in result.values() if isinstance(v,dict))
    FN = sum(v["total_answers"] - v[key] for v in result.values() if isinstance(v,dict))

    # TP = result[key]
    # FP = result["total_LLM_answers"]-result[key] #right??
    # FN = result["total_answers"]-result[key]
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    #check this still 
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


#This is the call used for the gold label checking

def Get_manual_evaluation_problems(print_results = False):
    problems, answers, problems_ex, answers_ex = get_LLM_problems(df, 60, set(), False, seed = 8)
    for problem in problems:
        #print(f"{problem}\n")
        print_example(df, ID = problem, rownum = None, ignore_highlights=True)
    if print_results: 
        print(len(problems))
    #print(f"problems: {problems}\n")
        print(f"answers{answers}\n")
    #print(f"problem: {problems_ex} \n answer: {answers_ex}\n")
    return problems, answers

def Get_prompts_for_LLM(amount = 10):
    problems, answers, problems_ex, answers_ex = get_LLM_problems(df, amount, set(), True)
    prompt_problems = {}
    ID_restore = {}
    for pairID in problems.keys():
        newkey = pairID[:-1]
        prompt_problems[newkey] = problems[pairID]
        ID_restore[newkey]=pairID[-1]

    step = max(1, math.ceil(amount / 10))  # every 10%
    prob = []

    with open("LLM_file.txt", "w") as f:
        keys = list(prompt_problems.keys())
        for idx, k in enumerate(keys):
            if idx % step == 0:
                f.write("----------------------------------\n")
            f.write(f"{k}:{prompt_problems[k]}\n")
            prob.append(f"{k}:{prompt_problems[k]}")

        #print(f"The example you can use is:\n{problems_ex}\n The answer:\n{answers_ex}")
        #print(f"The problems are:\n{prompt_problems}\n The answers are:{answers}")
        return ID_restore, answers, prob


def read_json(filename):
    """
    Function to read json files. 
    param filename: Name of the json file
    return data: json file in dict format
    """
    if not isinstance(filename, str):
        raise TypeError("Filename must be a string")

    if not filename.endswith(".json"):
        raise ValueError("The file given is not in a supported format (.json)")

    with open(filename, 'r') as json_file:
        data = json.load(json_file)
    return data



def checK_LLM(data, answers):
    if not isinstance(data, dict):
        raise TypeError("data must be a dictionary")

    # map short ids (no trailing label) to full ids (with trailing label)
    short_to_full = {pid[:-1]: pid for pid in answers.keys()}

    restored_data = {}
    for short_id, value in data.items():
        full_id = short_to_full.get(short_id)
        if full_id is not None:
            restored_data[full_id] = value
        # else: ignore entries not in gold set
    #Added count of misclassified types
    e = 0
    c = 0
    for pairID, pred in restored_data.items():
        label = pred.get("answer", "")
        if label == "entailment":
            e += 1
        elif label == "contradiction":
            c += 1
    perc_entailment = (e / (e + c) * 100) if (e + c) > 0 else 0.0

    result, scores_strict, scores_loose = check_LLM_answer(answers, restored_data)
    return result, scores_strict, scores_loose, perc_entailment




if __name__ == "__main__":
    

    #Get_manual_evaluation_problems(True)
    amount = 700
    ID_restore, answers, prob = Get_prompts_for_LLM(amount)


    LLM_answers_file = "LLM_answers_v1.json"
    LLM_answers = read_json(LLM_answers_file)
    result, strict, loose, perc_entailment = checK_LLM(LLM_answers, answers=answers)
    #print(f"LLM RESULTS: {result}")
    print("-----------------------------------")
    print(f"Strict scores: ", strict)
    print(f"Loose scores: ", loose)
    print(f"Percentage entailment", perc_entailment)


    # prompt_problems ={}
    # for pairID in problems.keys():
    #     newkey = pairID[:-1]
    #     prompt_problems[newkey] = problems[pairID]


    # for pairID in problems.keys():
    #     print("ANSWER")
    #     print(answers[pairID])
    #     print_example(df, ID = pairID)

    manual_LLM_answer_ex = {'pairID_0': ["man is a type of person", "black suit is a type of suit"]}



