from .common import *
from .abstract import AbstractPattern, StructuredExplanation

class RephrasingPattern(AbstractPattern):

    def __init__(self):
        self.patterns = {
            r"is a rephrasing of": "rephrasing",
            r"is rephrasing of": "rephrasing",
            r"is a rephrase of": "rephrase",
            r"is rephrase of": "rephrase",
            r"is a way to rephrase": "rephrase",
            r"is way to rephrase": "rephrase",
            r"is a way of saying": "saying",
            r"is way of saying": "saying",
            r"is short for": "short",
            r"is an abbreviation for": "abbreviation",
            r"is abbreviation for": "abbreviation",
            r"stanfor": "acronym",
        }
        self.relationship = '↔'

    def _generate_structured_explanation(self, anchor_word: str, doc: Doc, pattern_tokens: Span, highlights: List[str]) -> StructuredExplanation:

            # different anchor token for each pattern
            anchor_token = next((tok for tok in pattern_tokens if tok.text == anchor_word), None)
            if not anchor_token:
                raise ValueError(f"Anchor token not found in pattern")

            ancestors = list(anchor_token.ancestors)

            if ancestors:
                ancestor = ancestors[0]
                left_child = next((child for child in ancestor.lefts if not child.is_punct), None)

                if left_child:
                    left_term = list(left_child.subtree)
                else:

                    left_term = list(ancestor.lefts)
            else:
                # Fallback: anchor_token has no ancestors (root) -> collect left tokens up to punctuation
                left_term = AbstractPattern._get_left_tokens(anchor_token)

            # on the right, collect tokens in anchor_token's subtree, skipping the anchor itself
            # e.g. is a rephrasing of [something]
            right_term = AbstractPattern._get_right_tokens(anchor_token, pattern_tokens)

            left_string = " ".join(
                [tok.text for tok in left_term if tok.text not in ['"', "“", "”"]]+pattern_tokens.text.split()
            ).strip().lower()
            right_string = " ".join(
                [tok.text for tok in right_term if tok.text not in ['"', "“", "”"]]+pattern_tokens.text.split()
            ).strip().lower()

            left_string, right_string = AbstractPattern._get_grounded_terms(left_string, right_string, highlights)

            if not left_string or not right_string:
                raise ValueError("No grounded terms found")

            struct_expl = StructuredExplanation(self.relationship, [left_string, right_string], self.negate)
            return struct_expl

class ImplicationPattern(AbstractPattern):

    def __init__(self):
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
        self.patterns = AbstractPattern._generate_inflected_patterns(base_implication_patterns)
        self.relationship = '→'

    def _generate_structured_explanation(self, anchor_word: str, doc: Doc, pattern_tokens: Span, highlights: List[str]) -> StructuredExplanation:
            # anchor_token = next((tok for tok in pattern_tokens if tok.text.lower() == anchor_word.lower()), None)
            anchor_token = next((tok for tok in pattern_tokens if tok.text == anchor_word), None)
            if not anchor_token:
                # print(anchor_word, pattern_tokens)
                raise ValueError(f"Anchor token not found in pattern")

            # try to get ancestor -> spacy sometimes cannot find the ancestors of the anchor.
            # even if it cannot find the ancestor, there is a fallback mechanism that takes all the terms on the left
            ancestors = list(anchor_token.ancestors)
            if ancestors:
                ancestor = ancestors[0]
                left_child = next((child for child in ancestor.lefts if not child.is_punct), None)
                left_term = list(left_child.subtree) if left_child else list(ancestor.lefts)
            else:
                # fallback: use tokens to the left of the anchor if no ancestor
                left_term = [tok for tok in doc if tok.i < anchor_token.i]

            # get the right term
            right_term = AbstractPattern._get_right_tokens(anchor_token, pattern_tokens)

            left_string = " ".join([tok.text for tok in left_term if tok.text not in ['"', "“", "”"]+pattern_tokens.text.split()]).strip().lower()
            right_string = " ".join([tok.text for tok in right_term if tok.text not in ['"', "“", "”"]+pattern_tokens.text.split()]).strip().lower()

            left_string, right_string = AbstractPattern._get_grounded_terms(left_string, right_string, highlights)

            if not left_string or not right_string:
                raise ValueError("No grounded terms found")

            struct_expl = StructuredExplanation(self.relationship, [left_string, right_string], self.negate)
            return struct_expl

class EquivalencePattern(AbstractPattern):

    def __init__(self):
        self.patterns = {
        r"same as": "same",
        r"synonym of": "synonym",
        r"exchanged with": "exchanged",
        r"equivalent to": "equivalent",
    }
        self.relationship = '⊆'

    def _generate_structured_explanation(self, anchor_word: str, doc: Doc, pattern_tokens: Span, highlights: List[str]) -> StructuredExplanation:

            anchor_token = next((tok for tok in pattern_tokens if tok.text == anchor_word), None)
            if not anchor_token:
                raise ValueError(f"Anchor token not found in pattern")

            ancestors = list(anchor_token.ancestors)
            if ancestors:
                ancestor = ancestors[0]
                left_child = next((child for child in ancestor.lefts if not child.is_punct), None)
                if left_child:
                    left_term = list(left_child.subtree)
                else:
                    left_term = list(ancestor.lefts)
            else:
                # Fallback: anchor_token is root -> get tokens on the left until punctuation
                left_term = AbstractPattern._get_left_tokens(anchor_token)

            # Collect right side from anchor_token.subtree,
            # skipping the anchor itself and punctuation
            right_term = AbstractPattern._get_right_tokens(anchor_token, pattern_tokens)

            left_string = " ".join(
                tok.text for tok in left_term if tok.text not in ['"', "“", "”"]+pattern_tokens.text.split()
            ).strip().lower()
            right_string = " ".join(
                tok.text for tok in right_term if tok.text not in ['"', "“", "”"]+pattern_tokens.text.split()
            ).strip().lower()

            left_string, right_string = AbstractPattern._get_grounded_terms(left_string, right_string, highlights)

            if not left_string or not right_string:
                raise ValueError("No grounded terms found")

            struct_expl = StructuredExplanation(self.relationship, [left_string, right_string])
            return struct_expl

class IfThenPattern(AbstractPattern):

    def __init__(self):
        self.patterns = {
            r"then": "then",
        }
        self.relationship = '⇒'

    def _generate_structured_explanation(self, anchor_word: str, doc: Doc, pattern_tokens: Span, highlights: List[str]) -> StructuredExplanation:

        if_token = next((tok for tok in doc if tok.text.lower() == "if"), None)
        then_token = next((tok for tok in doc if tok.text.lower() == "then"), None)

        if not if_token or not then_token:
            raise ValueError("if or then not found")

        if if_token.i > then_token.i:
            raise ValueError("if is not before then")

        left_term = [
            tok for tok in doc[if_token.i + 1:then_token.i] if not tok.is_punct
        ]

        right_term = [
            tok for tok in doc[then_token.i + 1:] if not tok.is_punct
        ]

        left_string = " ".join(tok.text for tok in left_term).strip()
        right_string = " ".join(tok.text for tok in right_term).strip()

        left_string, right_string = AbstractPattern._get_grounded_terms(left_string, right_string, highlights)

        if not left_string or not right_string:
            raise ValueError("No grounded terms found")

        return StructuredExplanation(self.relationship, [left_string, right_string], self.negate)

class ClassificationPattern(AbstractPattern):

    def __init__(self):
        self.patterns = {
            r"type of": "type",
            r"kind of": "kind",
            r"sort of": "sort",
            r"form of": "form",
        }
        #self.patterns = AbstractPattern._generate_inflected_patterns(base_classification_patterns)
        self.relationship = '⊆'  # or "is-a"

    def _generate_structured_explanation(
        self, anchor_word: str, doc: Doc, pattern_tokens: Span, highlights: List[str]
    ) -> StructuredExplanation:

        anchor_token = next((tok for tok in pattern_tokens if tok.text == anchor_word), None)
        if not anchor_token:
                raise ValueError(f"Anchor token not found in pattern")

        ancestors = list(anchor_token.ancestors)
        if ancestors:
            ancestor = ancestors[0]
            left_child = next((child for child in ancestor.lefts if not child.is_punct), None)
            if left_child:
                left_term = list(left_child.subtree)
            else:
                left_term = list(ancestor.lefts)
        else:
            # Fallback: anchor_token is root -> get tokens on the left until punctuation
            left_term = AbstractPattern._get_left_tokens(anchor_token)

        # Right side: skip anchor itself from the subtree
        right_term = AbstractPattern._get_right_tokens(anchor_token, pattern_tokens)

        left_string = " ".join(tok.text for tok in left_term if tok.text not in ['"', "“", "”"]).strip()
        right_string = " ".join(tok.text for tok in right_term if tok.text not in ['"', "“", "”"]).strip()

        left_string, right_string = AbstractPattern._get_grounded_terms(left_string, right_string, highlights)

        if not left_string or not right_string:
            raise ValueError("No grounded terms found")

        return StructuredExplanation(self.relationship, [left_string, right_string], self.negate)

    def _find_additional_classifications(self, doc: Doc, highlights: List[str]) -> List[StructuredExplanation]:
        """
        checks for "X is a Y" classification by looking at the dependency parse
        """
        explanations = []
        for token in doc:
            # check for a root or main verb that is a form of 'be' (is, are, was, were, etc.)
            # also if it's the root (or at least a main verb).
            if token.lemma_ == "be" and token.dep_ == "ROOT":
                subj = [child for child in token.children if child.dep_ == "nsubj"]
                comp = [t for t in token.children if t.dep_ in ("attr", "acomp")]

                # if both exist and they are NOUN/PROPN, can consider it as classification
                if subj and comp:
                    if subj[0].pos_ in ("NOUN", "PROPN") and comp[0].pos_ in ("NOUN", "PROPN"):

                        comp_text = " ".join(tok.lemma_.lower() for tok in comp[0].subtree)
                        forbidden = {"rephrasing", "rephrase", "synonym", "equivalent", "type", "kind", "sort", "form", "same", "exchanged"}
                        if any(word in comp_text for word in forbidden):
                           continue

                        left_string = " ".join(tok.text for tok in subj[0].subtree)
                        right_string = " ".join(tok.text for tok in comp[0].subtree)

                        left_string, right_string = AbstractPattern._get_grounded_terms(left_string, right_string, highlights)

                        if not left_string or not right_string:
                            raise ValueError("No grounded terms found")

                        explanations.append(
                            StructuredExplanation(self.relationship, [left_string, right_string], self.negate)
                        )

        return explanations