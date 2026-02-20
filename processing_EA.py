import pandas as pd
import re
import string
"""This is our preprosessing script"""

def extract_ordered_highlighted_phrases(text, ordered = False):
        """
        Borrowed (and adapted) from previous project: 
        Scans a 'marked' sentence (e.g., "This church *choir* *sings* …")
        for highlighted substrings in order.
        If two highlights have only whitespace in between, they form
        a single phrase (e.g., "*cracks* *in* *the* *ceiling.*" -> "cracks in the ceiling").
        If there's intervening non-whitespace text, they're split
        into multiple items (e.g. "*masses* ... *book* ... *church.*" -> ["masses", "book", "church"]).

        Returns:
            A list of strings, each string representing one group of consecutive highlights.
        """
        if pd.isnull(text):
            return []

        # Use regex to find all segments enclosed in asterisks
        pattern = re.compile(r'\*(.*?)\*')
        matches = list(pattern.finditer(text))

        if ordered == True: 
            if not matches:
                return []

            phrases = []
            current_phrase = []
            translator = str.maketrans('', '', string.punctuation)

            for i, match in enumerate(matches):
                # The highlighted substring, e.g. "cracks"
                highlighted_text = match.group(1).strip()
                # Remove punctuation from the highlighted portion
                highlighted_text = highlighted_text.translate(translator).strip()
                if not highlighted_text:
                    # If it's empty after stripping, skip
                    continue

                if i == 0:
                    # first highlight: start a new phrase
                    current_phrase.append(highlighted_text)
                else:
                    # compare gap between previous match and current match
                    prev_end = matches[i-1].end()
                    curr_start = match.start()
                    in_between = text[prev_end:curr_start]

                    # If the gap is only whitespace, it's "consecutive highlights"
                    if in_between.strip() == '':
                        current_phrase.append(highlighted_text)
                    else:
                        # finish the old phrase, start a new one
                        phrases.append(" ".join(current_phrase))
                        current_phrase = [highlighted_text]

            # append the last phrase if any
            if current_phrase:
                phrases.append(" ".join(current_phrase))

            return phrases
        else: 
            return [match.group(1).strip(", ") for match in matches]
            #return [match.strip(", ") for match in matches]


def process_original(in_name, out_name):
    """
    Drop all columns that contain "Highlighted" as the authors state those columns should be ignored and the "marked" columns should be used instead. 
    For all three annotator sentences, create new columns containing the ordered highlights. 
    """
    df = pd.read_csv(in_name)
    #remove last character of each pairID
    #df["pairID"] = df["pairID"].astype(str).str[:-1]

    df = df.drop(
        columns=[col for col in df.columns if "Highlighted" in col]
    )

    for i in range(1, 4):
        col_s1_marked = f"Sentence1_marked_{i}"
        col_s2_marked = f"Sentence2_marked_{i}"

        col_s1_ordered = f"Sentence1_Highlighted_Ordered_{i}"
        col_s2_ordered = f"Sentence2_Highlighted_Ordered_{i}"

        df[col_s1_ordered] = df[col_s1_marked].apply(lambda txt: extract_ordered_highlighted_phrases(txt))
        df[col_s2_ordered] = df[col_s2_marked].apply(lambda txt: extract_ordered_highlighted_phrases(txt))

    df.to_csv(out_name)

def find_pattern(sentence, patterns):
    """
    Check whether any pattern occurs in a sentence.

    param: sentence (str): Input sentence.
    param: patterns (list): List of substrings to search for.
    returns: bool: True if any pattern is found, otherwise False.
    """
    for pattern in patterns:
        if pattern in sentence:
            return True 
    return False

def make_relevant_subset(in_name, out_name, labels, explanation_types = None, nr_matches =2 ):
    """ 
    From an already preprocessed dataset, get the relevant subset that we want to work with. 
    Keeps only the problems with a label specified in labels 
    Based on the classification type it keeps the problems for which some annotators (amount determined by nr_matches) use one of the specified keywords.
    Currently only works for explanation_types = "classification" but can be expanded if needed.
    A new column "explanation_type"  is added where the explanation type is stored. (currently only classification)
    A new column "matching_explanations" is added  where the numbers of the annotators who's explanations matched one of the keywords are stored. 
    """
    df = pd.read_csv(in_name)
    ex_patterns = {}
    if "classification" in explanation_types:
        ex_patterns["classification"] = ["type of", "kind of", "form of" ]
    df = df[df["gold_label"].isin(labels)].copy()#.reset_index(drop=True)
    df["explanation_type"] = None
    df["matching_explanations"] = None 
  
    for ex_type, patterns in ex_patterns.items():

        matches = pd.DataFrame({
            f"Explanation_{i}": df[f"Explanation_{i}"].apply(lambda x: find_pattern(str(x), patterns))
            for i in range(1, 4)
        })

        # Count how many explanations match per row
        match_count = matches.sum(axis=1)

        # Assign explanation_type if enough explanations match
        mask = match_count >= nr_matches
        df.loc[mask, "explanation_type"] = ex_type

        # Create column listing which explanations matched
        df.loc[mask, "matching_explanations"] = matches[mask].apply(
            lambda row: ",".join(str(i) for i, matched in enumerate(row, start=1) if matched),
            axis=1
        )
    
    # Keep only rows where an explanation_type was assigned
    df = df[df["explanation_type"].notna()]
    df.to_csv(out_name)


#process_original("esnli_dev.csv", "processed_esnli_EA.csv")
#make_relevant_subset("processed_esnli_EA.csv", "entailment_probs_2.csv", ["entailment"], ["classification"], 2)
#make_relevant_subset("processed_esnli_EA.csv", "entailment_probs_or.csv", ["entailment"], ["classification"], 1)
#process_original("esnli_test.csv", "processed_esnli_test_EA.csv")
#make_relevant_subset("processed_esnli_test_EA.csv", "entailment_probs_test_or.csv", ["entailment"], ["classification"], 1)
#make_relevant_subset("processed_esnli_EA.csv", "entailment_probs.csv", ["entailment"], None)

dev= pd.read_csv("esnli_dev.csv")
test =pd.read_csv("esnli_test.csv")

print(len(dev.index), len(test.index))




