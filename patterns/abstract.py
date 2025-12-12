from __future__ import annotations
from .common import *

@dataclass(frozen=True)
class StructuredExplanation():
    '''
    Recursively represent a Structured explanation as a relationship between n predicates where a predicate can be
    a phrase or another explanation
    '''

    relationship: str
    predicates: Tuple[Union[str, "StructuredExplanation"], ...]
    negated: bool = False

    def __str__(self):
        rep = (" " + self.relationship + " ").join([str(p) for p in self.predicates])
        if self.negated:
            return f'¬({rep})'
        return rep

    def __bool__(self):
        return bool(self.relationship) and len(self.predicates) > 0

    def __eq__(self, other: object) -> bool:
        """Recursively check both structure and leaves, handling commutativity."""
        if not isinstance(other, StructuredExplanation):
            return False

        if self.relationship != other.relationship or self.negated != other.negated:
            return False

        if len(self.predicates) != len(other.predicates):
            return False

        # Handle commutative operations by sorting predicates before comparison
        if self.relationship == "∧":  
            return sorted(self.predicates, key=str) == sorted(other.predicates, key=str)

        # Otherwise, preserve order
        return all(sp == op for sp, op in zip(self.predicates, other.predicates))


class AbstractPattern(ABC):

    patterns: Dict[str, str]
    relationship: str
    negate: bool = False

    def __call__(self, doc: Doc, highlights: List[str]) -> List[StructuredExplanation]:
        '''
        contains main loop for generating structured explanations using the patterns
        of the class

        @param doc: parsed string
        @return: StructuredExplanation object
        '''
        pattern_tokens = self._find_pattern_tokens(doc)
        explanations = []
        for toks in pattern_tokens:
            try:
                anchor_word = self.patterns.get(str(toks), None)
                explanations.append(self._generate_structured_explanation(anchor_word, doc, toks, highlights))
            except (IndexError, StopIteration, ValueError) as e:
                continue
                # print(f"Error processing: {doc.text}\nError: {e}")
        extra_expl = self._find_additional_classifications(doc, highlights)
        explanations.extend(extra_expl)
        return AbstractPattern.concatenate_explanations(explanations)

    @abstractmethod
    def _generate_structured_explanation(self, anchor_word: str, doc: Doc, pattern_tokens: Span, highlights: List[str]) -> StructuredExplanation:
        raise NotImplementedError

    def _find_additional_classifications(self, doc: Doc, highlights: List[str]) -> List[StructuredExplanation]:
        return []

    def _find_pattern_tokens(self, doc: Doc):
        ## matching all pattern at once and return all spans where it matched
        matches = re.findall(r'|'.join(self.patterns), str(doc))
        toks = []
        for i, _ in enumerate(doc):
            if not matches:
                break
            for m in matches:
                w_len = len(m.split())
                if str(doc[i:i+w_len]) == m:
                    toks.append(doc[i:i+w_len])
                    matches.remove(m)
                    break
        return toks

    def _generate_inflected_patterns(base_patterns):
        expanded_patterns = {}

        for pattern, _ in base_patterns.items():
            words = pattern.split()

            # first find the verb in the phrase
            verb_token = next((token for token in nlp(pattern) if token.pos_ == "VERB"), None)

            if not verb_token:
                expanded_patterns[pattern] = pattern
                continue

            # get the pre words and post words of the verb and keep them for later use
            pre_words = pattern.split(verb_token.text)[0].strip()
            post_words = pattern.split(verb_token.text)[-1].strip()

            # generate inflected forms of the verb
            inflected_forms = [
                verb_token._.inflect("VB"),   # Base form
                verb_token._.inflect("VBZ"),  # Third-person singular
                verb_token._.inflect("VBG"),  # Present participle
                verb_token._.inflect("VBD"),  # Past tense
                verb_token._.inflect("VBN"),  # Past participle
            ]

            all_forms = list(filter(None, inflected_forms))

            for form in all_forms:
                # generate the full phrase with pre words, different form of the verb and post words
                full_phrase = f"{pre_words} {form} {post_words}".strip()
                expanded_patterns[full_phrase] = form

                # "can + verb" form
                can_phrase = f"{pre_words} can {form} {post_words}".strip()
                expanded_patterns[can_phrase] = form
                could_phrase = f"{pre_words} could {form} {post_words}".strip()
                expanded_patterns[could_phrase] = form

        return expanded_patterns

    def _generate_negative_patterns(base_patterns):
        negative_patterns = {}

        for pattern, base_form in base_patterns.items():
            doc = nlp(pattern)
            verb_token = next((token for token in doc if token.pos_ == "VERB"), None)

            if not verb_token:
                negative_patterns[f"does not {pattern}"] = base_form
                negative_patterns[f"doesn't {pattern}"] = base_form
                negative_patterns[f"did not {pattern}"] = base_form
                negative_patterns[f"didn't {pattern}"] = base_form
                negative_patterns[f"cannot {pattern}"] = base_form
                negative_patterns[f"can not {pattern}"] = base_form
                negative_patterns[f"can't {pattern}"] = base_form
                continue

            base_verb = verb_token._.inflect("VB") or verb_token.text

            pre_words = pattern.split(verb_token.text)[0].strip()
            post_words = pattern.split(verb_token.text)[-1].strip()

            neg_do_not = f"{pre_words} do not {base_verb} {post_words}".strip()  # "do not imply"
            neg_dont = f"{pre_words} don't {base_verb} {post_words}".strip()  # "don't imply"
            neg_does_not = f"{pre_words} does not {base_verb} {post_words}".strip()  # "does not imply"
            neg_doesnt = f"{pre_words} doesn't {base_verb} {post_words}".strip()  # "doesn't imply"
            neg_did_not = f"{pre_words} did not {base_verb} {post_words}".strip()  # "did not imply"
            neg_didnt = f"{pre_words} didn't {base_verb} {post_words}".strip()  # "didn't imply"
            neg_cannot = f"{pre_words} cannot {base_verb} {post_words}".strip()  # "cannot imply"
            neg_can_not = f"{pre_words} can not {base_verb} {post_words}".strip()  # "can not imply"
            neg_cant = f"{pre_words} can't {base_verb} {post_words}".strip()  # "can't imply"
            neg_couldnot = f"{pre_words} couldnot {base_verb} {post_words}".strip()  # "couldnot imply"
            neg_could_not = f"{pre_words} could not {base_verb} {post_words}".strip()  # "could not imply"
            neg_couldnt = f"{pre_words} could't {base_verb} {post_words}".strip()  # "couldn't imply"

            negative_patterns[neg_do_not] = base_form
            negative_patterns[neg_dont] = base_form
            negative_patterns[neg_does_not] = base_form
            negative_patterns[neg_doesnt] = base_form
            negative_patterns[neg_did_not] = base_form
            negative_patterns[neg_didnt] = base_form
            negative_patterns[neg_cannot] = base_form
            negative_patterns[neg_can_not] = base_form
            negative_patterns[neg_cant] = base_form
            negative_patterns[neg_couldnot] = base_form
            negative_patterns[neg_could_not] = base_form
            negative_patterns[neg_couldnt] = base_form

            present_participle = verb_token._.inflect("VBG")
            if present_participle:
                neg_not_ing = f"{pre_words} not {present_participle} {post_words}".strip()  # "not implying"
                negative_patterns[neg_not_ing] = base_form

        return negative_patterns


    def _get_left_tokens(anchor_token: spacy.tokens.token.Token) -> List[spacy.tokens.token.Token]:

        # anchor_token.sent gives the sentence span in which `anchor_token` is located
        sent = anchor_token.sent
        # only consider the part of the sentence before the anchor token
        left_span = sent[: anchor_token.i - sent.start]

        # collect tokens in reverse, then reverse them back,
        # stopping if we see punctuation.
        collected = []
        for tok in reversed(left_span):
            if tok.is_punct or tok.dep_ == "punct":
                break
            if tok.pos_ in ("CCONJ", "SCONJ"):
                break
            collected.append(tok)

        collected.reverse()
        return collected

    def _get_right_tokens(anchor_token: spacy.tokens.token.Token, pattern_tokens: Span) -> List[spacy.tokens.token.Token]:
        collected = []
        first_token_skipped = False

        raw_right_subtree = list(anchor_token.subtree)[1:]
        right_tokens = [
                t for t in raw_right_subtree
                if t.text not in [tok.text for tok in pattern_tokens]
                and t.text not in ['"', '“', '”']]

        for tok in right_tokens:
            if not first_token_skipped and tok.text.lower() in ["that", "of", "as"]:
                first_token_skipped = True
                continue

            first_token_skipped = True

            if tok.is_punct or tok.dep_ == "punct":
                break
            if tok.pos_ in ("CCONJ", "SCONJ"):
                break

            collected.append(tok)

        return collected

    def _get_grounded_terms(left: str, right: str, highlights: List[str]):

        highlights = sorted(highlights, key=len, reverse=True) ## try to match longer highlights first
        grounded_left = left
        grounded_right = right
        for term in highlights:
            if grounded_left == left and re.search(term, left, re.IGNORECASE):
                grounded_left = term
            if grounded_right == right and re.search(term, right, re.IGNORECASE):
                grounded_right = term

            if grounded_right != right and grounded_left != left:
                break
        return grounded_left, grounded_right


    def concatenate_explanations(expls: List[StructuredExplanation]) -> StructuredExplanation:
        if len(expls) == 0:
            return StructuredExplanation('', [])
        if len(expls) == 1:
            return expls[0]

        concat = StructuredExplanation('∧', expls[0:2])
        for expl in expls[2:]:
            concat = StructuredExplanation('∧', [concat, expl])

        return concat

    def _token_to_text(tokens: List[Doc]) -> str:
        return " ".join([token.text for token in tokens])