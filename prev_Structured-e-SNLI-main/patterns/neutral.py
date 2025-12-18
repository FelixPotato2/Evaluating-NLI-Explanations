from .common import *
from .abstract import AbstractPattern, StructuredExplanation
from .entailment import *

class NeutralImplicationPattern(ImplicationPattern):

    def __init__(self):
        super().__init__()

        base_implication_patterns = {
            r"imply that": "imply",
            r"imply": "imply",
            r"necessarily imply": "imply",
            r"suggest that": "suggest",
            r"suggest": "suggest",
            r"indicate that": "indicate",
            r"indicate": "indicate",
            r"result in": "result",
            r"entail that": "entail",
            r"entail": "entail",
            r"infer": "infer",
            r"infer as": "infer",
            r"mean": "mean",
            r"necessarily mean": "mean",
        }

        self.patterns = AbstractPattern._generate_negative_patterns(base_implication_patterns)
        # self.relationship = '⊭'
        self.negate = True

class NotAllPattern(AbstractPattern):

    def __init__(self):
        super().__init__()
        self.patterns = {
            'are': 'are'
        }
        self.relationship = '⊉'


    def _generate_structured_explanation(self, anchor_word: str, doc: Doc, pattern_tokens: Span, highlights: List[str]) -> StructuredExplanation:
        not_token = next((tok for tok in doc if tok.text.lower() == "not"), None)
        all_token = next((tok for tok in doc if tok.text.lower() == "all" and tok.i == (not_token.i + 1)), None) if not_token else None
        are_token = next((tok for tok in doc if tok.text.lower() == "are"), None)

        if not not_token or not all_token or not are_token:
            raise ValueError("not, all, and are not found")

        if all_token.i > are_token.i:
            raise ValueError("not, all, and are not found")

        left_term = [
            tok for tok in doc[all_token.i + 1:are_token.i] if not tok.is_punct
        ]

        right_term = [
            tok for tok in doc[are_token.i + 1:] if not tok.is_punct
        ]

        left_string = " ".join(tok.text for tok in left_term).strip()
        right_string = " ".join(tok.text for tok in right_term).strip()

        left_string, right_string = AbstractPattern._get_grounded_terms(left_string, right_string, highlights)

        if not left_string or not right_string:
            raise ValueError("No grounded terms found")

        return StructuredExplanation(self.relationship, [left_string, right_string])
