import pandas as pd
import re
import string

df = pd.read_csv("entailment_probs_2.csv")

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
    example : bool, optional
        If True, also extract the first row of df as an example
        for use in an LLM prompt.

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
    available_df = df[~df["pairID"].isin(excluded_ids)]
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

        df = df.drop(0)
    else:
        pair_dict_ex = None
        answer_dict_ex = None

    # Sample problems
    sampled_df = available_df.sample(n=nr_problems, random_state=seed)

    pair_dict = {
        row["pairID"]: {
            "premise": row["Sentence1"],
            "hypothesis": row["Sentence2"],
        }
        for i, row in sampled_df.iterrows()
    }
    excluded_ids.update(pair_dict.keys())

    answer_dict, missing_answers = get_correct_answers(sampled_df)
    #handle missing answers
    if missing_answers:
        sampled_df = sampled_df[~sampled_df["pairID"].isin(missing_answers)]
        pair_dict2, answer_dict2, _, _ =get_LLM_problems(available_df,  len(missing_answers), excluded_ids, example= False, seed = seed+1)
        pair_dict.update(pair_dict2)
        answer_dict.update(answer_dict2)
    return pair_dict, answer_dict, pair_dict_ex, answer_dict_ex

def get_correct_answers(df):
    """
    Randomly sample problems from a DataFrame.

    Parameters:
    df : pandas.DataFrame
        Input dataframe with sampled problems.

    Returns:
    answers_dict : dict
        Dictionary with for each pairID a list of answers
        The list of answers contains dictionaries with left and right side of answer and annotator number
    """
    answers_dict = {}
    missing_answers =[]

    for idx, row in df.iterrows():
        ann_matches = row["matching_explanations"].split(",")
        answers = []

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
                left = splitted_ex[j].split()[-4:] 
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
                    grouped = False
                    #group answer with previous answers if they overlap
                    #the grouping makes it even worse if there are words in the left or right part of answer that shouldnt be there
                    for answer in answers:
                        if bool(set(w for group in answer["left"] for w in group) & set(left_overlap)) or bool(set(w for group in answer["right"] for w in group)):
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
                #signal that a correct answer could not be extracted with the highlights
                else: 
                    missing_answers.append(row["pairID"])

        answers_dict[row["pairID"]] = answers

    return answers_dict, missing_answers


def get_overlap(text, highlighted):
    """
    Gets from the text the parts that were highlighted
    
    Parameters: 
    text: left or right side of explanation
    highlighted: highlighted words from ESNLI explanation
    
    Returns:
    result: list of the words in the text that were also highlighted (excluding articles)
    """
    result = []

    for word in text:
        cleaned_word = word.strip(" ,.").lower()

        if cleaned_word in {"a", "an", "the"}:
            continue

        if cleaned_word in highlighted:
            result.append(cleaned_word)
    
    #if result > 1:
        #todo, check if they are next to eachother, otherwise delete the most far one 

    return result





problems, answers, problems_ex, answers_ex = get_LLM_problems(df, 5, set(), True)
print(f"problems: {problems}\n")
print(f"answers{answers}\n")
print(f"problem: {problems_ex} \n answer: {answers_ex}\n")

manual_LLM_answer_ex = {'pairID_0': ["man is a type of person", "black suit is a type of suit"]}
#Todo
"""write function that given a dict with problems and everything generates an LLM prompt with all the problems in that dict """