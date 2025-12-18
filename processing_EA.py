import pandas as pd

def extract_ordered_highlighted_phrases(text):
        """
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


def process_original(in_name, out_name):
    df = pd.read_csv(in_name)
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
    given a sentence and a list of patterns return whether the sentence 
    """
    for pattern in patterns:
        if pattern in sentence:
            return True 
    return False



def make_relevant_subset(in_name, out_name, labels, explanation_types = None, nr_matches =2 ):
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

        # Assign explanation_type if at least 2 explanations match
        #df.loc[match_count >= nr_matches, "explanation_type"] = ex_type

         # Assign explanation_type if at least `min_matches` explanations match
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


make_relevant_subset("processed_esnli_EA.csv", "entailment_probs_2.csv", ["entailment"], ["classification"], 2)
#make_relevant_subset("processed_esnli_EA.csv", "entailment_probs.csv", ["entailment"], None)
#process_original("esnli_dev.csv", "processed_esnli_EA.csv")



