import pandas as pd
import re
from explore_esnli_data import print_example
import json
import ast
import math

df = pd.read_csv("merged_entailment.csv")
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
                r"(?:type of|form of|kind of)",
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

def calculate_scores(result, key):
    TP = sum(v[key] for v in result.values() if isinstance(v, dict))
    FP = sum(v["total_LLM_answers"] - v[key] for v in result.values() if isinstance(v,dict))
    FN = sum(v["total_answers"] - v[key] for v in result.values() if isinstance(v,dict))

    # TP = result[key]
    # FP = result["total_LLM_answers"]-result[key] #right??
    # FN = result["total_answers"]-result[key]
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    #check this still 
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}

def calculate_counts(result, key):
    """Return TP/FP/FN totals for debugging/analysis."""
    TP = sum(v[key] for v in result.values() if isinstance(v, dict))
    FP = sum(v["total_LLM_answers"] - v[key] for v in result.values() if isinstance(v, dict))
    FN = sum(v["total_answers"] - v[key] for v in result.values() if isinstance(v, dict))
    return {"TP": TP, "FP": FP, "FN": FN}

def subset_with_extra_limit(pred_tokens, gold_tokens, max_extra=2):
    pred_set = set(pred_tokens)
    gold_set = set(gold_tokens)

    if not gold_set.issubset(pred_set):
        return False, None

    extras = len(pred_set - gold_set)
    return extras <= max_extra, extras

def _extract_typeof_relations(explanations):
    """Return only explanations that contain 'is a type of'."""
    if not explanations:
        return []
    out = []
    for ex in explanations:
        if isinstance(ex, str) and "is a type of" in ex.lower():
            out.append(ex)
    return out

def check_LLM_answer(answers, LLM_output, max_extra_total=2):
    """
    Adds a new metric (len_ok) where an LLM relation is correct if:
      - gold-left ⊆ llm-left AND gold-right ⊆ llm-right
      - (extra_left + extra_right) <= max_extra_total
    Exact matches override len_ok, which overrides partial.
    """
    pairs_with_no_typeof = 0
    pairs_with_no_output = 0

    result = {}
    for pairID in answers:
        if pairID not in LLM_output:
            result[pairID] = {
                "exact": 0,
                "partial": 0,
                "len_ok": 0,
                "combined_correct": 0,          # exact + partial 
                "combined_len_ok": 0,           # exact + len_ok 
                "total_answers": len(answers[pairID]),
                "total_LLM_answers": 0,
            }
            pairs_with_no_output += 1
            continue

        exact_count = 0
        partial_count = 0
        len_ok_count = 0


        # llm_explanations = LLM_output[pairID].get("explanation", [])
        raw_explanations = LLM_output[pairID].get("explanation", [])
        llm_explanations = _extract_typeof_relations(raw_explanations)

        for LLM_answer in llm_explanations:
            if len(llm_explanations) == 0:
                pairs_with_no_typeof += 1
            LLM_answer = (LLM_answer or "").lower() 
            splitted_ans = LLM_answer.split("is a type of")
            if len(splitted_ans) < 2:
                continue

            articles = {"a", "an", "the"}
            # spl_str_ans becomes: [left_tokens, right_tokens]
            spl_str_ans = [
                [tok.strip(" ,.") for tok in seg.lower().split()
                 if tok.strip(" ,.").lower() not in articles]
                for seg in splitted_ans[:2]
            ]

            llm_left = spl_str_ans[0]
            llm_right = spl_str_ans[1]

            found_exact = False
            found_len_ok = False
            found_partial = False

            for answer_group_dict in answers[pairID]:
                left_exact = False
                right_exact = False
                left_partial = False
                right_partial = False

                # For len_ok we need combined extras across sides.
                # We'll collect extras for all gold lefts that are subsets, and all gold rights that are subsets.
                left_extras_candidates = []
                right_extras_candidates = []

                # Left side checks
                for left_answer in answer_group_dict.get("left", []):
                    if left_answer == llm_left:
                        left_exact = True
                    if bool(set(left_answer) & set(llm_left)):
                        left_partial = True

                    okL, extraL = subset_with_extra_limit(llm_left, left_answer, max_extra=max_extra_total)
                    if okL:
                        left_extras_candidates.append(extraL)

                # Right side checks
                for right_answer in answer_group_dict.get("right", []):
                    if right_answer == llm_right:
                        right_exact = True
                    if bool(set(right_answer) & set(llm_right)):
                        right_partial = True

                    okR, extraR = subset_with_extra_limit(llm_right, right_answer, max_extra=max_extra_total)
                    if okR:
                        right_extras_candidates.append(extraR)

                # Decide exact for this group
                if left_exact and right_exact:
                    found_exact = True
                    break

                # Decide len_ok for this group: any pairing with total extras <= max_extra_total
                if left_extras_candidates and right_extras_candidates:
                    # smallest possible total extras across any (left,right) pairing
                    if (min(left_extras_candidates) + min(right_extras_candidates)) <= max_extra_total:
                        found_len_ok = True

                # Decide partial for this group
                if left_partial and right_partial:
                    found_partial = True

            # Count at most once per LLM relation (exact > len_ok > partial)
            if found_exact:
                exact_count += 1
            elif found_len_ok:
                len_ok_count += 1
            elif found_partial:
                partial_count += 1

        result[pairID] = {
            "exact": exact_count,
            "partial": partial_count,
            "len_ok": len_ok_count,
            "combined_correct": exact_count + partial_count,
            "combined_len_ok": exact_count + len_ok_count,
            "total_answers": len(answers[pairID]),
            "total_LLM_answers": len(llm_explanations),
        }

    scores_strict = calculate_scores(result, "exact")
    scores_loose = calculate_scores(result, "combined_correct")
    len_scores = calculate_scores(result, "combined_len_ok")
    len_metrics = calculate_counts(result, "combined_len_ok")

    print('Pairs with no type of: ', pairs_with_no_typeof)
    print('Pairs with empty explanation: ', pairs_with_no_output)

    return result, scores_strict, scores_loose, len_scores, len_metrics

def checK_LLM(data, answers, id_map):
    if not isinstance(data, dict):
        raise TypeError("data must be a dictionary")

    # Restore shortened IDs
    restored_data = {}
    for sid, value in data.items():
        full_id = id_map.get(sid)
        if full_id is not None:
            restored_data[full_id] = value
    
    # after restoring ids
    parsed_ids = set(restored_data.keys())

    # evaluate only on ids that were successfully parsed
    answers = {pid: answers[pid] for pid in answers if pid in parsed_ids}
    
    # Count predicted labels (basic check)
    e = 0
    c = 0
    for _, pred in restored_data.items():
        label = (pred.get("answer", "") or "").strip().lower()
        if label == "entailment":
            e += 1
        elif label == "contradiction":
            c += 1
    perc_entailment = (e / (e + c) * 100) if (e + c) > 0 else 0.0

    result, strict, loose, len_scores, len_metrics = check_LLM_answer(answers, restored_data)

    return result, strict, loose, perc_entailment, len_metrics, len_scores

#This is the call used for the gold label checking

def Get_manual_evaluation_problems(print_results = True):
    problems, answers, problems_ex, answers_ex = get_LLM_problems(dev_df, 60, set(), False, seed = 8)
    if print_results: 
        for problem in problems:
            #print(f"{problem}\n")
            print_example(df, ID = problem, rownum = None, ignore_highlights=True)
        print(len(problems))
        #print(f"problems: {problems}\n")
        print(f"answers{answers}\n")
        #print(f"problem: {problems_ex} \n answer: {answers_ex}\n")
    return problems, answers

def Get_prompts_for_LLM(amount=10):
    problems, answers, problems_ex, answers_ex = get_LLM_problems(df, amount, set(), True)
    with open("annotators_answers.json", "w", encoding="utf-8") as aa:
        json.dump(answers, aa, ensure_ascii=False, indent=2)

    id_map = {}
    prompt_problems = {}

    full_ids = list(problems.keys())

    for idx, full_id in enumerate(full_ids, start=1):
        sid = f"q{idx:04d}"   
        id_map[sid] = full_id
        prompt_problems[sid] = problems[full_id]

    step = max(1, math.ceil(amount / 10))
    prob = []

    with open("LLM_file.txt", "w") as f:
        keys = list(prompt_problems.keys())
        for idx, k in enumerate(keys):
            if idx % step == 0:
                f.write("----------------------------------\n")
            f.write(f"{k}:{prompt_problems[k]}\n")
            prob.append(f"{k}:{prompt_problems[k]}")

    with open("id_map.json", "w", encoding="utf-8") as f:
        json.dump(id_map, f, ensure_ascii=False, indent=2)

    return id_map, answers, prob

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


if __name__ == "__main__":

    dev_df = pd.read_csv("entailment_probs_or.csv")
    df = pd.read_csv("merged_entailment.csv")

    LLM_answers_file = "final_LLM_auto_responses.json"
    LLM_answers = read_json(LLM_answers_file)

    answers = read_json("annotators_answers.json")
    id_map = read_json("id_map.json")

    result, strict, loose, perc_entailment, len_metrics, len_scores = checK_LLM(
        LLM_answers,
        answers=answers,
        id_map=id_map
    )
    print('Len Scores: ', len_scores)
    print('Len metrics: ', len_metrics)
    print("Loose ", loose)