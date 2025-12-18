import re
import string
import pandas as pd
import spacy 

class ESNLIPreprocessor:
    def __init__(self, data, csv_path=""):
        """
        Initialize the processor by loading the dataset from a CSV file.

        Args:
            data (pd.DataFrame): DataFrame containing the e-SNLI dataset.
            csv_path (str): Path to the e-SNLI dataset in CSV format. For now it's empty.
        """
        self.data = data
        self.cleaned_data = pd.DataFrame()
        self.nlp = spacy.load("en_core_web_sm")

    def extract_highlighted_words(self):
        """
        Extract highlighted words for Sentence1 and Sentence2 for all explanations into separate columns.

        Returns:
            pd.DataFrame: DataFrame with extracted highlighted words.
        """
        for i in range(1, 4):  # Iterate through all 3 explanations
            self.data[f'Sentence1_Highlighted_Words_{i}'] = self.data.apply(
                lambda row: self._parse_highlighted_words(row[f'Sentence1_Highlighted_{i}'], row['Sentence1']), axis=1
            ).apply(self._remove_punctuation)
            self.data[f'Sentence2_Highlighted_Words_{i}'] = self.data.apply(
                lambda row: self._parse_highlighted_words(row[f'Sentence2_Highlighted_{i}'], row['Sentence2']), axis=1
            ).apply(self._remove_punctuation)

        return self.data

    def _extract_ordered_highlighted_phrases(self, text):
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


    def create_ordered_highlights_as_list(self):
        """
        Creates new columns for each 'marked' sentence in i=1..3:
            sentence1_highlighted_ordered_i, sentence2_highlighted_ordered_i
        Each column will be a list of strings, where each string represents
        one group of consecutive highlighted words.
        """
        for i in range(1, 4):
            col_s1_marked = f"Sentence1_marked_{i}"
            col_s2_marked = f"Sentence2_marked_{i}"

            col_s1_ordered = f"Sentence1_Highlighted_Ordered_{i}"
            col_s2_ordered = f"Sentence2_Highlighted_Ordered_{i}"

            self.data[col_s1_ordered] = self.data[col_s1_marked].apply(
                lambda txt: self._extract_ordered_highlighted_phrases(txt)
            )
            self.data[col_s2_ordered] = self.data[col_s2_marked].apply(
                lambda txt: self._extract_ordered_highlighted_phrases(txt)
            )

        return self.data



    def cleanup_and_restructure(self):
        """
        Create a new cleaned and restructured DataFrame with only relevant columns.

        Returns:
            pd.DataFrame: Cleaned and restructured DataFrame.
        """
        # Define columns to drop
        columns_to_drop = []
        for i in range(1, 4):
            columns_to_drop.extend([
                f'Sentence1_marked_{i}',
                f'Sentence2_marked_{i}',
                f'Sentence1_Highlighted_{i}',
                f'Sentence2_Highlighted_{i}'
            ])

        self.cleaned_data = self.data.drop(columns=columns_to_drop, errors='ignore')

        # Define the new column order
        columns_order = ['pairID', 'gold_label',
                        'Sentence1', 'Sentence2', 'Sentence1_Length', 'Sentence2_Length',
                        'Explanation_1', 'Sentence1_Highlighted_Ordered_1', 'Sentence2_Highlighted_Ordered_1','Sentence1_Highlighted_Words_1', 'Sentence2_Highlighted_Words_1',
                        'Sentence1_Highlight_Count_1', 'Sentence2_Highlight_Count_1',
                        #'Sentence1_Highlighted_Lemmas_1', 'Sentence2_Highlighted_Lemmas_1', 'Sentence1_Highlighted_POS_1', 'Sentence2_Highlighted_POS_1',
                        'Explanation_2', 'Sentence1_Highlighted_Ordered_2', 'Sentence2_Highlighted_Ordered_2', 'Sentence1_Highlighted_Words_2', 'Sentence2_Highlighted_Words_2',
                        'Sentence1_Highlight_Count_2', 'Sentence2_Highlight_Count_2',
                        #'Sentence1_Highlighted_Lemmas_2', 'Sentence2_Highlighted_Lemmas_2', 'Sentence1_Highlighted_POS_2', 'Sentence2_Highlighted_POS_2',
                        'Explanation_3', 'Sentence1_Highlighted_Ordered_3', 'Sentence2_Highlighted_Ordered_3', 'Sentence1_Highlighted_Words_3', 'Sentence2_Highlighted_Words_3',
                        'Sentence1_Highlight_Count_3', 'Sentence2_Highlight_Count_3',
                        #'Sentence1_Highlighted_Lemmas_3', 'Sentence2_Highlighted_Lemmas_3', 'Sentence1_Highlighted_POS_3', 'Sentence2_Highlighted_POS_3',
                        ]

        # Reorder columns
        self.cleaned_data = self.cleaned_data[columns_order]
        return self.cleaned_data


    def add_sentence_lengths(self):
        """
        Add sentence lengths for Sentence1 and Sentence2.
        """
        self.data["Sentence1_Length"] = self.data["Sentence1"].apply(lambda x: len(x.split()))
        self.data["Sentence2_Length"] = self.data["Sentence2"].apply(lambda x: len(x.split()))
        return self.data

    def count_highlighted_words(self):
        """
        Count the number of highlighted words for Sentence1 and Sentence2 for all explanations.
        """
        for i in range(1, 4):
            self.data[f"Sentence1_Highlight_Count_{i}"] = self.data[f"Sentence1_Highlighted_Words_{i}"].apply(
                lambda x: len(x) if isinstance(x, list) else 0
            )
            self.data[f"Sentence2_Highlight_Count_{i}"] = self.data[f"Sentence2_Highlighted_Words_{i}"].apply(
                lambda x: len(x) if isinstance(x, list) else 0
            )
        return self.data

    def lemmatize_highlighted_words(self):
        """
        Perform lemmatization and POS tagging on highlighted words and store results in separate columns.
        """
        for i in range(1, 4):
            self.data[[f"Sentence1_Highlighted_Lemmas_{i}", f"Sentence1_Highlighted_POS_{i}"]] = self.data.apply(
                lambda row: self._lemmatize_and_pos(row['Sentence1'], row[f'Sentence1_Highlighted_Words_{i}']),
                axis=1, result_type="expand"
            )
            self.data[[f"Sentence2_Highlighted_Lemmas_{i}", f"Sentence2_Highlighted_POS_{i}"]] = self.data.apply(
                lambda row: self._lemmatize_and_pos(row['Sentence2'], row[f'Sentence2_Highlighted_Words_{i}']),
                axis=1, result_type="expand"
            )
        return self.data

    @staticmethod
    def _parse_highlighted_words(highlighted_indices, sentence):
        """
        Parse the highlighted words from indices in a column and map them to the words in the sentence.

        Args:
            highlighted_indices (str): String representation of highlighted indices.
            sentence (str): The sentence from which the highlighted words are extracted.

        Returns:
            list: List of highlighted words.
        """
        try:
            if pd.isnull(highlighted_indices) or pd.isnull(sentence):
                return []
            indices = [int(idx) for idx in highlighted_indices.strip("{} ").split(",")]
            words = sentence.split()
            return [words[idx] for idx in indices if 0 <= idx < len(words)]
        except Exception as e:
            return []

    @staticmethod
    def _remove_punctuation(word_list):
        """
        Remove punctuation from a list of words.

        Args:
            word_list (list): List of words.

        Returns:
            list: List of words with punctuation removed.
        """
        translator = str.maketrans('', '', string.punctuation)
        return [word.translate(translator) for word in word_list]

    def _lemmatize_and_pos(self, sentence, highlighted_words):
        """
        Lemmatize highlighted words and extract their POS tags using sentence context.

        Args:
            sentence (str): The entire sentence for context.
            highlighted_words (list): List of highlighted words to process.

        Returns:
            tuple: Two lists - (lemmas, pos_tags).
        """
        if not isinstance(highlighted_words, list) or len(highlighted_words) == 0:
            return [], []

        doc = self.nlp(sentence)
        lemmas = []
        pos_tags = []

        for token in doc:
            if token.text in highlighted_words:
                lemmas.append(token.lemma_)
                pos_tags.append(token.pos_)

        return lemmas, pos_tags
