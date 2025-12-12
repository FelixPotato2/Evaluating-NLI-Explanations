from .abstract import AbstractPattern
from .entailment import *

class NotRephrasingPattern(RephrasingPattern):
    def __init__(self):
        super().__init__()
        self.patterns = {
            ' '.join(k.split()[0:1] + ['not'] + k.split()[1:]): v
            for k, v in self.patterns.items()
        }
        self.negate = True

class NotImplicationPattern(ImplicationPattern):
    def __init__(self):
        super().__init__()
        base_implication_patterns = {
            r"imply that": "imply",
            r"imply": "imply",
            r"suggest that": "suggest",
            r"suggest": "suggest",
            r"indicate that": "indicate",
            r"indicate": "indicate",
            r"result in": "result",
            r"entail that": "entail",
            r"entail": "entail",
            r"infer": "infer",
            r"infer as": "infer",
            r"mean": "mean"
        }

        self.negate = True

        self.patterns = AbstractPattern._generate_negative_patterns(base_implication_patterns)

class NotEquivalencePattern(EquivalencePattern):
    def __init__(self):
        super().__init__()
        self.patterns = {
            ' '.join( ['not'] + k.split()): v
            for k, v in self.patterns.items()
        }
        self.relationship = '⊈'

class NotClassificationPattern(ClassificationPattern):
    def __init__(self):
        super().__init__()
        self.patterns = {
            ' '.join(['not'] + k.split()): v
            for k, v in self.patterns.items()
        }
        self.relationship = '⊈'

class XORPattern(AbstractPattern):

    def __init__(self):
        self.patterns = {
            r"or": "or"
        }
        self.relationship = '⊕'

    def _generate_structured_explanation(self, anchor_word: str, doc: Doc, pattern_tokens: Span, highlights: List[str]) -> StructuredExplanation:

        either_token = next((tok for tok in doc if tok.text.lower() == "either"), None)
        or_token = next((tok for tok in doc if tok.text.lower() == "or"), None)

        if not either_token or not or_token:
            return None

        if either_token.i > or_token.i:
            return None

        left_term = [
            tok for tok in doc[either_token.i + 1:or_token.i] if not tok.is_punct
        ]

        right_term = [
            tok for tok in doc[or_token.i + 1:] if not tok.is_punct
        ]

        left_string = " ".join(tok.text for tok in left_term).strip()
        right_string = " ".join(tok.text for tok in right_term).strip()

        left_string, right_string = AbstractPattern._get_grounded_terms(left_string, right_string, highlights)

        if not left_string or not right_string:
            raise ValueError("No grounded terms found")

        return StructuredExplanation(self.relationship, (left_string, right_string))

class CannotBePattern(AbstractPattern):
    def __init__(self):
        self.patterns = {
            r"can not be": "cannot_be",
            r"cannot be": "cannot_be",
            r"can't be": "cannot_be",
        }
        self.relationship = '⊕'
        self.split_words = {"while", "and", "or", "but"}

    def _find_pattern_tokens(self, doc: Doc):
        regex = re.compile(r'|'.join(self.patterns.keys()), re.IGNORECASE)
        matches = list(regex.finditer(doc.text))
        spans = []
        for match in matches:
            span = doc.char_span(match.start(), match.end())
            if span is not None:
                spans.append(span)
        return spans

    def _generate_structured_explanation(
        self, anchor_word: str, doc: Doc, pattern_tokens: Span, highlights: List[str]
    ) -> StructuredExplanation:

        pattern_end = pattern_tokens[-1].i
        tokens_after = list(doc[pattern_end + 1:])

        # find a splitting token (e.g. "while", "and", etc.)
        split_index = None
        for idx, tok in enumerate(tokens_after):
            if tok.text.lower() in self.split_words:
                split_index = idx
                break

        if split_index is not None and split_index > 0:
            # tokens before the split word form the left term
            left_tokens = tokens_after[:split_index]
            # tokens after the splitting word form the right term
            right_tokens = tokens_after[split_index + 1:]
        else:
            # if no splitting token found in the tokens after the pattern
            # try to use tokens before the pattern as the left term
            if pattern_tokens[0].i > 0:
                left_context = list(doc[:pattern_tokens[0].i])
                left_tokens = left_context[-3:] if len(left_context) >= 3 else left_context
            else:
                left_tokens = []
            right_tokens = tokens_after

        left_string = " ".join(tok.text for tok in left_tokens).strip()
        right_string = " ".join(tok.text for tok in right_tokens if not tok.is_punct).strip()

        left_string, right_string = AbstractPattern._get_grounded_terms(left_string, right_string, highlights)

        if not left_string or not right_string:
            raise ValueError("No grounded terms found for 'cannot be' pattern.")

        return StructuredExplanation(self.relationship, (left_string, right_string))
