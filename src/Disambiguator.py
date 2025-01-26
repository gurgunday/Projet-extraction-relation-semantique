from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Set
from Token import Token, Pos
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DisambiguatedToken:
    word: str
    lemma: str
    pos: Pos
    additional_info: Dict[str, str]
    weight: float


class Disambiguator:
    STRONG_ADVS = {"très", "trop", "plus", "moins", "assez", "si", "tout"}
    NEG_MARKERS = {"ne", "n'"}
    NEG_COMPLEMENTS = {
        "pas",
        "plus",
        "jamais",
        "point",
        "rien",
        "personne",
        "aucun",
        "aucune",
        "nullement",
    }
    COMMON_PREPOSITIONS = {
        "à",
        "au",
        "aux",
        "de",
        "des",
        "du",
        "pour",
        "par",
        "en",
        "dans",
        "sur",
        "sous",
        "avec",
        "vers",
        "chez",
        "devant",
        "derrière",
        "entre",
        "parmi",
    }
    MODAL_VERBS = {"pouvoir", "devoir", "vouloir", "falloir", "savoir"}
    AUXILIARY_VERBS = {"être", "avoir"}
    QUESTION_WORDS = {
        "que",
        "qu'",
        "qui",
        "où",
        "quand",
        "comment",
        "pourquoi",
        "quel",
        "quelle",
        "quels",
        "quelles",
        "quoi",
        "lequel",
        "laquelle",
        "lesquels",
        "lesquelles",
        "combien",
    }
    SUBJECT_PRONOUNS = {
        "je",
        "j'",
        "tu",
        "il",
        "elle",
        "on",
        "nous",
        "vous",
        "ils",
        "elles",
        "celui",
        "celle",
        "ceux",
        "celles",
    }
    DETERMINERS = {
        "le",
        "la",
        "les",
        "un",
        "une",
        "des",
        "du",
        "de",
        "d'",
        "mon",
        "ton",
        "son",
        "ma",
        "ta",
        "sa",
        "mes",
        "tes",
        "ses",
        "notre",
        "votre",
        "leur",
        "nos",
        "vos",
        "leurs",
        "ce",
        "cet",
        "cette",
        "ces",
        "quelque",
        "quelques",
        "chaque",
        "tout",
        "toute",
        "tous",
        "toutes",
    }
    CONJUNCTIONS = {
        "et",
        "ou",
        "mais",
        "donc",
        "car",
        "ni",
        "or",
        "puis",
        "ensuite",
        "enfin",
        "pourtant",
        "cependant",
        "néanmoins",
        "toutefois",
        "ainsi",
        "alors",
    }
    DEMONSTRATIVE_PRONOUNS = {
        "ce",
        "c'",
        "ça",
        "ceci",
        "cela",
        "celui",
        "celle",
        "ceux",
        "celles",
        "celui-ci",
        "celle-ci",
        "ceux-ci",
        "celles-ci",
        "celui-là",
        "celle-là",
        "ceux-là",
        "celles-là",
    }
    RELATIVE_PRONOUNS = {
        "qui",
        "que",
        "qu'",
        "dont",
        "où",
        "lequel",
        "laquelle",
        "lesquels",
        "lesquelles",
        "duquel",
        "de laquelle",
        "desquels",
        "desquelles",
        "auquel",
        "à laquelle",
        "auxquels",
        "auxquelles",
    }
    VERB_ENDINGS = {
        "er",
        "ir",
        "re",  # Infinitive
        "é",
        "i",
        "u",
        "t",  # Past participle
        "ant",  # Present participle
        "ais",
        "ait",
        "ions",
        "iez",
        "aient",  # Imperfect
        "ai",
        "as",
        "a",
        "ons",
        "ez",
        "ent",  # Present
        "erai",
        "eras",
        "era",
        "erons",
        "erez",
        "eront",  # Future
    }

    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.token_count = len(tokens)
        self.pos_sequence = []
        self.context_window = 3 

    def _map_pos_tag(self, tag: str) -> Optional[Pos]:
        """Enhanced POS tag mapping with detailed categories"""
        mapping = {
            "Nom": Pos.NOM,
            "Ver": Pos.VER,
            "Adj": Pos.ADJ,
            "Adv": Pos.ADV,
            "Det": Pos.DET,
            "Pro": Pos.PRO,
            "Pre": Pos.PRE,
            "Conj": Pos.CONJ,
            "Punct": Pos.PUNCT,
            "PROrel": Pos.PRO, 
            "PROdem": Pos.PRO,
            "PROpers": Pos.PRO,
            "DETpos": Pos.DET,
            "DETdem": Pos.DET,
            "CONJcoo": Pos.CONJ,
            "CONJsub": Pos.CONJ,
        }
        return mapping.get(tag, None)

    def _extract_info_from_pos_info(self, pos_info) -> Dict[str, str]:
        """Enhanced feature extraction with comprehensive patterns"""
        info = {}

        for additional in pos_info.additional_info:
            parts = additional.split("+")
            for part in parts:
                # Gender
                if part in ["Mas", "Fem"]:
                    info["Gender"] = part
                # Number
                elif part == "SG":
                    info["Number"] = "Sing"
                elif part == "PL":
                    info["Number"] = "Plur"
                # Type information
                elif part in [
                    "Qual",
                    "Def",
                    "Ind",
                    "Pers",
                    "Dem",
                    "Poss",
                    "Int",
                    "Rel",
                ]:
                    info["Type"] = part
                # Person
                elif part in ["P1", "P2", "P3"]:
                    info["Person"] = part
                # Verb tenses
                elif "Pre" in part:
                    if "IPre" in part:
                        info["Tense"] = "Present"
                        info["Mode"] = "Indicatif"
                    elif "SPre" in part:
                        info["Tense"] = "Present"
                        info["Mode"] = "Subjonctif"
                    elif "CPre" in part:
                        info["Tense"] = "Present"
                        info["Mode"] = "Conditionnel"
                    elif "ImPre" in part:
                        info["Tense"] = "Present"
                        info["Mode"] = "Impératif"
                elif "Imp" in part:
                    info["Tense"] = "Imparfait"
                    info["Mode"] = "Indicatif"
                elif "Fut" in part:
                    info["Tense"] = "Futur"
                    info["Mode"] = "Indicatif"
                elif "PQP" in part:
                    info["Tense"] = "PlusQueParfait"
                    info["Mode"] = "Indicatif"
                elif part == "Inf":
                    info["Form"] = "Infinitive"
                elif part == "PPas":
                    info["Form"] = "PastParticiple"
                elif part == "PPre":
                    info["Form"] = "PresentParticiple"

        return info

    def _is_proper_noun(self, token: Token, position: int) -> bool:
        """Enhanced proper noun detection"""
        word = token.word

        # Check for typical proper noun patterns
        if (
            word[0].isupper()
            and position > 0
            and not any(pos.startswith("Ver") for pos in token.all_pos_tags)
        ):

            # Additional checks for false positives
            if len(word) > 1:
                if not word.lower() in self.QUESTION_WORDS: 
                    if not word.lower() in self.SUBJECT_PRONOUNS:  # Not pronoun
                        if not any(
                            word.lower().endswith(end) for end in self.VERB_ENDINGS
                        ):  # Not verb
                            return True
        return False

    def _is_likely_verb(self, token: Token, position: int) -> bool:
        """Enhanced verb pattern detection"""
        word_lower = token.word.lower()

        if any(word_lower.endswith(ending) for ending in self.VERB_ENDINGS):
            return True

        if position > 0:
            prev_token = self.tokens[position - 1]
            prev_word = prev_token.word.lower()

            if (
                prev_word in self.SUBJECT_PRONOUNS
                or prev_word in self.MODAL_VERBS
                or prev_word in self.NEG_MARKERS
            ):
                return True

        # Check following token for verb complements
        if position + 1 < self.token_count:
            next_token = self.tokens[position + 1]
            next_word = next_token.word.lower()

            if next_word in self.NEG_COMPLEMENTS:
                return True

        return False

    def _get_contextual_weight(self, token: Token, pos: str, position: int) -> float:
        base_weight = 0.5
        word_lower = token.word.lower()

        # Look back for negation markers within a reasonable window
        if word_lower == "pas" and pos == "Adv":
            # Look back up to 3 words for ne/n'
            for i in range(max(0, position - 3), position):
                if self.tokens[i].word.lower() in ["ne", "n'"]:
                    return 2.0

        if position > 0:
            prev_token = self.tokens[position - 1]
            prev_pos_tags = prev_token.all_pos_tags

            # After determiner
            if "Det" in prev_pos_tags:
                if pos == "Nom":
                    return 2.0  # Strongly prefer noun after determiner
                if pos == "Ver":
                    return 0.2  # Significantly lower weight for verb after determiner

            # After noun subject
            if "Nom" in prev_pos_tags:
                if pos == "Ver":
                    verbal_info = [
                        info for info in token._word_pos_info if info.pos == "Ver"
                    ]
                    if any(
                        "IPre" in " ".join(info.additional_info) for info in verbal_info
                    ):
                        return 2.0  # Prefer conjugated verbs after noun subjects

        # At start of sentence
        if position == 0:
            if pos == "Det" and word_lower in self.DETERMINERS:
                return 1.8  # Prefer determiners at start

        return base_weight

    def _get_candidates(
        self, token: Token, position: int
    ) -> List[Tuple[Pos, float, Dict[str, str]]]:
        """Enhanced POS candidate generation"""
        candidates = []
        word_lower = token.word.lower()

        if word_lower in {"c'", "l'", "d'", "n'", "j'", "s'", "qu'"}:
            if word_lower == "c'" and position + 1 < self.token_count:
                next_word = self.tokens[position + 1].word.lower()
                if next_word == "est":
                    return [(Pos.PRO, 2.0, {"Type": "Dem"})]
            elif word_lower == "n'":
                return [(Pos.NEG, 2.0, {})]
            elif word_lower in {"l'", "d'"}:
                return [(Pos.DET, 2.0, {"Type": "Def"})]

        if word_lower in self.AUXILIARY_VERBS:
            return [(Pos.VER, 2.0, {"Type": "Aux"})]

        if self._is_proper_noun(token, position):
            return [(Pos.NOM, 2.0, {"Type": "Proper"})]

        for pos_info in token._word_pos_info:
            if pos_info.pos in ["Gender", "Number", "Verbal"]:
                continue

            pos = self._map_pos_tag(pos_info.pos)
            if not pos:
                continue

            weight = self._get_contextual_weight(token, pos_info.pos, position)
            info = self._extract_info_from_pos_info(pos_info)

            if self._is_likely_verb(token, position) and pos == Pos.VER:
                weight *= 1.5
            if word_lower in self.COMMON_PREPOSITIONS and pos == Pos.PRE:
                weight *= 1.5
            if word_lower in self.DETERMINERS and pos == Pos.DET:
                weight *= 1.5

            candidates.append((pos, weight, info))

        return candidates if candidates else [(Pos.UNK, 0.1, {})]

    def _adjust_by_context(
        self,
        candidates: List[Tuple[Pos, float, Dict[str, str]]],
        position: int,
        prev_token: Optional[DisambiguatedToken] = None,
    ) -> List[Tuple[Pos, float, Dict[str, str]]]:
        """Enhanced contextual adjustment with improved agreement handling"""
        if not candidates:
            return []

        word = self.tokens[position].word.lower()
        adjusted = []

        for pos, weight, info in candidates:
            adj_weight = weight

            # Agreement patterns
            if prev_token:
                # Determiner-Noun agreement
                if prev_token.pos == Pos.DET and pos in {Pos.NOM, Pos.ADJ}:
                    adj_weight *= 1.2
                    # Propagate agreement features
                    for attr in ["Gender", "Number"]:
                        if attr in prev_token.additional_info:
                            info[attr] = prev_token.additional_info[attr]

                # Subject-Verb agreement
                if pos == Pos.VER:
                    if prev_token.pos == Pos.NOM:
                        info["Person"] = "P3"
                        info["Number"] = prev_token.additional_info.get(
                            "Number", "Sing"
                        )
                        adj_weight *= 1.2
                    elif (
                        prev_token.pos == Pos.PRO
                        and "Person" in prev_token.additional_info
                    ):
                        info["Person"] = prev_token.additional_info["Person"]
                        info["Number"] = prev_token.additional_info.get(
                            "Number", "Sing"
                        )
                        adj_weight *= 1.2

                # Adjective-Noun agreement
                if pos == Pos.ADJ and prev_token.pos == Pos.NOM:
                    for attr in ["Gender", "Number"]:
                        if attr in prev_token.additional_info:
                            info[attr] = prev_token.additional_info[attr]
                    adj_weight *= 1.2

                # Negation patterns
                if prev_token.word.lower() in self.NEG_MARKERS:
                    if pos == Pos.VER:
                        adj_weight *= 1.3
                    elif word in self.NEG_COMPLEMENTS and pos == Pos.ADV:
                        adj_weight *= 1.5
                        info["Type"] = "Neg"

                # Prefer 'pas' as adverb when preceded by a verb
                if word == "pas" and prev_token.pos == Pos.VER:
                    if pos == Pos.ADV:
                        adj_weight *= (
                            2.0
                        )
                    elif pos == Pos.VER:
                        adj_weight *= (
                            0.2
                        )

                # Modal verb constructions
                if prev_token.lemma in self.MODAL_VERBS:
                    if pos == Pos.VER and info.get("Form") == "Infinitive":
                        adj_weight *= 1.5

                # Preposition patterns
                if prev_token.pos == Pos.PRE:
                    if pos in {Pos.NOM, Pos.DET}:
                        adj_weight *= 1.2

            # Look ahead for additional context
            if position + 1 < self.token_count:
                next_token = self.tokens[position + 1]
                next_word = next_token.word.lower()

                # Verb-Object patterns
                if pos == Pos.VER and next_word in self.DETERMINERS:
                    adj_weight *= 1.2

                # Adjective post-position
                if pos == Pos.NOM and next_word in self.COMMON_PREPOSITIONS:
                    adj_weight *= 1.2

            adjusted.append((pos, adj_weight, info))

        return adjusted

    def disambiguate(self) -> List[DisambiguatedToken]:
        """Main disambiguation process with enhanced context handling"""
        results = []
        context_window = 3  # Look at previous N tokens for context

        for position, token in enumerate(self.tokens):
            candidates = self._get_candidates(token, position)

            context_start = max(0, position - context_window)
            prev_tokens = results[context_start:position]
            prev_token = prev_tokens[-1] if prev_tokens else None

            candidates = self._adjust_by_context(candidates, position, prev_token)

            if not candidates:
                results.append(
                    DisambiguatedToken(token.word, token.word, Pos.UNK, {}, 0.0)
                )
                continue

            candidates.sort(key=lambda x: x[1], reverse=True)
            best_pos, weight, info = candidates[0]

            lemma = (
                token.word
                if position == 0 and token.word[0].isupper()
                else token.word.lower()
            )

            results.append(
                DisambiguatedToken(token.word, lemma, best_pos, info, weight)
            )

        return results

    def initial_disambiguation(self) -> List[DisambiguatedToken]:
        """Initial disambiguation pass"""
        return self.disambiguate()

    def disambiguate_with_feedback(
        self, feedback: Dict[int, str]
    ) -> List[DisambiguatedToken]:
        """Disambiguation with feedback incorporation"""
        # Get initial disambiguation
        tokens = self.disambiguate()

        # If we have feedback, adjust the problematic tokens
        if feedback:
            for position, error in feedback.items():
                if position < len(tokens):
                    token = self.tokens[position]
                    candidates = self._get_candidates(token, position)

                    # Filter candidates based on feedback
                    if "unresolved" in error:
                        # Try to avoid the problematic POS
                        problematic_pos = error.split("_")[1].upper()
                        candidates = [
                            (pos, w, i)
                            for pos, w, i in candidates
                            if pos.name != problematic_pos
                        ]

                    if candidates:
                        # Choose the next best candidate
                        candidates.sort(key=lambda x: x[1], reverse=True)
                        best_pos, weight, info = candidates[0]
                        tokens[position] = DisambiguatedToken(
                            token.word, token.word.lower(), best_pos, info, weight
                        )

        return tokens


if __name__ == "__main__":
    from Loader import WebLoader
    from Tokenizer import Tokenizer
    from Token import Token
    import logging

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    def test_and_print(sentence: str, web_loader: WebLoader, expected_pos=None):
        """Test a sentence and print detailed results with validation"""
        print(f"\nTesting sentence: {sentence}")
        print("=" * 80)

        # Get tokens
        tokens = Tokenizer(sentence, web_loader, None)

        # Create disambiguator and get results
        disambiguator = Disambiguator(tokens)
        results = disambiguator.disambiguate()

        # Print disambiguation results
        all_correct = True
        for i, result in enumerate(results):
            print(f"\nToken {i+1}: {result.word}")
            print(f"{'':4}POS: {result.pos.name}")
            print(f"{'':4}Lemma: {result.lemma}")
            print(f"{'':4}Weight: {result.weight:.2f}")

            # Print additional info in a more structured way
            if result.additional_info:
                print(f"{'':4}Features:")
                for key, value in result.additional_info.items():
                    print(f"{'':8}{key}: {value}")

            # Validate against expected POS if provided
            if expected_pos and i < len(expected_pos):
                if result.pos != expected_pos[i]:
                    print(
                        f"{'':4}WARNING: Expected {expected_pos[i].name}, got {result.pos.name}"
                    )
                    all_correct = False

            print(f"token: {tokens[i]}")

        if expected_pos:
            print(f"\nPOS Validation: {'PASSED' if all_correct else 'FAILED'}")
        print("\n" + "-" * 80)

    # Initialize loader
    web_loader = WebLoader("")

    print("\nRunning Enhanced Disambiguation Tests")
    print("=" * 80)

    # 1. Preposition Tests
    print("\n=== Preposition Tests ===")
    preposition_tests = [
        ("Il va à Paris", [Pos.PRO, Pos.VER, Pos.PRE, Pos.NOM]),  # 'à' as preposition
        (
            "Je parle de mon chat",
            [Pos.PRO, Pos.VER, Pos.PRE, Pos.DET, Pos.NOM],
        ),  # 'de' as preposition
        (
            "Il travaille avec passion",
            [Pos.PRO, Pos.VER, Pos.PRE, Pos.NOM],
        ),  # 'avec' as preposition
    ]
    for sentence, expected in preposition_tests:
        test_and_print(sentence, web_loader, expected)

    # 2. Proper Noun Tests
    print("\n=== Proper Noun Tests ===")
    proper_noun_tests = [
        ("Marie aime Pierre", [Pos.NOM, Pos.VER, Pos.NOM]),  # Both proper nouns
        ("Paris est belle", [Pos.NOM, Pos.VER, Pos.ADJ]),  # City name
        (
            "Le petit Pierre dort",
            [Pos.DET, Pos.ADJ, Pos.NOM, Pos.VER],
        ),  # Proper noun with adjective
    ]
    for sentence, expected in proper_noun_tests:
        test_and_print(sentence, web_loader, expected)

    # 3. Verbal Construction Tests
    print("\n=== Verbal Construction Tests ===")
    verb_tests = [
        ("Je peux partir", [Pos.PRO, Pos.VER, Pos.VER]),  # Modal + infinitive
        ("Il a mangé", [Pos.PRO, Pos.VER, Pos.VER]),  # Auxiliary + participle
        ("Elle donne le livre", [Pos.PRO, Pos.VER, Pos.DET, Pos.NOM]),  # Simple present
    ]
    for sentence, expected in verb_tests:
        test_and_print(sentence, web_loader, expected)

    # 4. Negation Pattern Tests
    print("\n=== Negation Pattern Tests ===")
    negation_tests = [
        ("Je ne mange pas", [Pos.PRO, Pos.ADV, Pos.VER, Pos.ADV]),
        (
            "Il n'aime plus le chocolat",
            [Pos.PRO, Pos.ADV, Pos.VER, Pos.ADV, Pos.DET, Pos.NOM],
        ),
        ("Tu ne peux jamais dormir", [Pos.PRO, Pos.ADV, Pos.VER, Pos.ADV, Pos.VER]),
    ]
    for sentence, expected in negation_tests:
        test_and_print(sentence, web_loader, expected)

    # 5. Complex Agreement Tests
    print("\n=== Complex Agreement Tests ===")
    agreement_tests = [
        (
            "Les petites filles dorment",
            [Pos.DET, Pos.ADJ, Pos.NOM, Pos.VER],
        ),  # Plural feminine agreement
        (
            "Cette belle maison bleue",
            [Pos.DET, Pos.ADJ, Pos.NOM, Pos.ADJ],
        ),  # Multiple adjective agreement
        (
            "Mes grands chats noirs",
            [Pos.DET, Pos.ADJ, Pos.NOM, Pos.ADJ],
        ),  # Plural masculine agreement
    ]
    for sentence, expected in agreement_tests:
        test_and_print(sentence, web_loader, expected)

    # 6. Ambiguous Word Tests
    print("\n=== Ambiguous Word Tests ===")
    ambiguous_tests = [
        (
            "La ferme est grande",
            [Pos.DET, Pos.NOM, Pos.VER, Pos.ADJ],
        ),  # 'ferme' as noun
        ("Il ferme la porte", [Pos.PRO, Pos.VER, Pos.DET, Pos.NOM]),  # 'ferme' as verb
        (
            "Le livre est sur la table",
            [Pos.DET, Pos.NOM, Pos.VER, Pos.PRE, Pos.DET, Pos.NOM],
        ),  # 'livre' as noun
    ]
    for sentence, expected in ambiguous_tests:
        test_and_print(sentence, web_loader, expected)

    # 7. Question Pattern Tests
    print("\n=== Question Pattern Tests ===")
    question_tests = [
        ("Que mange le chat", [Pos.PRO, Pos.VER, Pos.DET, Pos.NOM]),
        ("Où vas-tu", [Pos.PRO, Pos.VER, Pos.PRO]),
        ("Comment allez-vous", [Pos.ADV, Pos.VER, Pos.PRO]),
    ]
    for sentence, expected in question_tests:
        test_and_print(sentence, web_loader, expected)

    # 8. Special Construction Tests
    print("\n=== Special Construction Tests ===")
    special_tests = [
        ("C'est un chat", [Pos.PRO, Pos.VER, Pos.DET, Pos.NOM]),  # Contraction
        ("L'homme parle", [Pos.DET, Pos.NOM, Pos.VER]),  # Elision
        ("Vas-y maintenant", [Pos.VER, Pos.PRO, Pos.ADV]),  # Imperative with pronoun
    ]
    for sentence, expected in special_tests:
        test_and_print(sentence, web_loader, expected)
