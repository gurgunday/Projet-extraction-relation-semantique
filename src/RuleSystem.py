from enum import Enum, auto
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass
from Token import Pos
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@dataclass
class Rule:
    name: str
    pattern: List[Pos]
    result: Pos
    priority: int = 1

    def __str__(self):
        pattern_str = " + ".join(pos.name for pos in self.pattern)
        return f"{self.name}: {pattern_str} → {self.result.name}"


@dataclass
class ParsedNode:
    word: str
    lemma: str
    pos: Pos
    weight: float
    additional_info: Dict[str, str]
    start: int
    end: int
    children: List["ParsedNode"] = None

    def __post_init__(self):
        if self.children is None:
            self.children = []

    def __str__(self):
        return f"{self.word}({self.pos.name})"

    def propagate_agreement(self):
        if not self.children:
            return

        # Features to propagate
        agreement_features = {"Gender", "Number", "Person"}

        # Get features from parent
        features_to_propagate = {
            feature: value
            for feature, value in self.additional_info.items()
            if feature in agreement_features
        }

        # Propagate to children
        for child in self.children:
            child.additional_info.update(features_to_propagate)
            child.propagate_agreement()


class ParsingError(Exception):
    def __init__(self, feedback: Dict[int, str]):
        self.feedback = feedback
        super().__init__(f"Failed to parse sentence. Feedback: {feedback}")


RULES = [
    # 1. Passive voice rules
    Rule("VER + VER", [Pos.VER, Pos.VER], Pos.GV, priority=3),  # "est mangée"
    Rule("GV + GP", [Pos.GV, Pos.GP], Pos.GV, priority=2),  # "est mangée par le chat"
    Rule(
        "DET + NOM", [Pos.DET, Pos.NOM], Pos.GNDet, priority=2
    ),  # "la souris", "le chat"
    Rule("PRE + GNDet", [Pos.PRE, Pos.GNDet], Pos.GP, priority=2),  # "par le chat"
    Rule("GNDet + GV", [Pos.GNDet, Pos.GV], Pos.P, priority=1),  # Full passive sentence
    # 1. Adjective 
    Rule("Adjective + Noun", [Pos.ADJ, Pos.NOM], Pos.GN, priority=3),
    Rule("Noun + Adjective", [Pos.NOM, Pos.ADJ], Pos.GN, priority=3),
    Rule("GN + Adjective", [Pos.GN, Pos.ADJ], Pos.GN, priority=3),
    # 2. Determiner 
    Rule("Determiner + Noun", [Pos.DET, Pos.NOM], Pos.GNDet, priority=2),
    Rule("Determiner + GN", [Pos.DET, Pos.GN], Pos.GNDet, priority=2),
    Rule("GNDet + Adjective", [Pos.GNDet, Pos.ADJ], Pos.GNDet, priority=2),
    # 3. Negation rules 
    Rule("NEG + VER", [Pos.NEG, Pos.VER], Pos.GVNeg, priority=4),
    Rule(
        "NEG + VER + ADV", [Pos.NEG, Pos.VER, Pos.ADV], Pos.GVNeg, priority=4
    ),  # New rule for "ne...pas"
    Rule("GVNeg + GNDet", [Pos.GVNeg, Pos.GNDet], Pos.GVNeg, priority=3),
    # 4. Verb group rules
    Rule("VER + ADV", [Pos.VER, Pos.ADV], Pos.GV, priority=2),
    Rule("VER + GNDet", [Pos.VER, Pos.GNDet], Pos.GV, priority=2),
    Rule("GV + GNDet", [Pos.GV, Pos.GNDet], Pos.GV, priority=2),
    Rule("GV + ADV", [Pos.GV, Pos.ADV], Pos.GV, priority=2),
    Rule("VER + GP", [Pos.VER, Pos.GP], Pos.GV, priority=2),
    Rule("GV + GP", [Pos.GV, Pos.GP], Pos.GV, priority=2),
    # 5. Prepositional phrases
    Rule("PRE + GNDet", [Pos.PRE, Pos.GNDet], Pos.GP, priority=2),
    Rule("PRE + NOM", [Pos.PRE, Pos.NOM], Pos.GP, priority=2),
    # 6. Complete sentences
    Rule(
        "PRO + GVNeg", [Pos.PRO, Pos.GVNeg], Pos.P, priority=3
    ),  # High priority for negative sentences
    Rule("PRO + GV", [Pos.PRO, Pos.GV], Pos.P, priority=1),
    Rule("GNDet + GV", [Pos.GNDet, Pos.GV], Pos.P, priority=1),
    Rule("GNDet + GVNeg", [Pos.GNDet, Pos.GVNeg], Pos.P, priority=1),
    Rule("GNDet + VER", [Pos.GNDet, Pos.VER], Pos.P, priority=1),
    Rule("P + GP", [Pos.P, Pos.GP], Pos.P, priority=1),
    Rule("P + PUNCT", [Pos.P, Pos.PUNCT], Pos.P, priority=1),
]


class RuleSystem:
    def __init__(self, rules: List[Rule] = RULES):
        self.rules = sorted(rules, key=lambda x: x.priority, reverse=True)
        logger.info(f"Initialized RuleSystem with {len(rules)} rules")

    def _check_agreement(self, nodes: List[ParsedNode]) -> bool:
        """Check if nodes agree in gender, number, and person"""
        if not nodes:
            return True

        # Get features from first node
        gender = nodes[0].additional_info.get("Gender")
        number = nodes[0].additional_info.get("Number")
        person = nodes[0].additional_info.get("Person")

        # Check agreement across all nodes
        for node in nodes[1:]:
            if "Gender" in node.additional_info and gender:
                if node.additional_info["Gender"] != gender:
                    return False
            if "Number" in node.additional_info and number:
                if node.additional_info["Number"] != number:
                    return False
            if "Person" in node.additional_info and person:
                if node.additional_info["Person"] != person:
                    return False

        return True

    def _preprocess_negative_particles(
        self, tokens: List[ParsedNode]
    ) -> List[ParsedNode]:
        """Enhanced negative particle preprocessing with support for complex patterns"""
        processed_tokens = []
        i = 0
        while i < len(tokens):
            current_token = tokens[i]

            # Handle "ne/n'" ... "pas/plus/jamais/point" pattern
            if current_token.word.lower() in ["ne", "n'"] and i + 1 < len(tokens):

                # Look ahead for negative complement
                next_tokens = tokens[i + 1 :]
                complement_index = -1

                for j, token in enumerate(next_tokens):
                    if token.word.lower() in ["pas", "plus", "jamais", "point"]:
                        complement_index = i + 1 + j
                        break

                if complement_index != -1:
                    # Create negative construction
                    included_tokens = tokens[i : complement_index + 1]
                    neg_node = ParsedNode(
                        word=" ".join(t.word for t in included_tokens),
                        lemma="ne",
                        pos=Pos.GVNeg,
                        weight=current_token.weight,
                        additional_info={
                            "negation": "complete",
                            "Person": included_tokens[1].additional_info.get("Person"),
                            "Number": included_tokens[1].additional_info.get("Number"),
                        },
                        start=current_token.start,
                        end=tokens[complement_index].end,
                        children=included_tokens,
                    )
                    processed_tokens.append(neg_node)
                    i = complement_index + 1
                    continue

            processed_tokens.append(current_token)
            i += 1

        return processed_tokens

    def _match_rule(self, tokens: List[ParsedNode], pattern: List[Pos]) -> bool:
        """Enhanced pattern matching with agreement checking"""
        if len(tokens) != len(pattern):
            return False

        # Check POS tags match
        if not all(
            token.pos == expected_pos for token, expected_pos in zip(tokens, pattern)
        ):
            return False

        # Check agreement
        return self._check_agreement(tokens)

    def _apply_rule(self, tokens: List[ParsedNode], rule: Rule) -> ParsedNode:
        """Enhanced rule application with improved feature handling"""
        # Combine words with proper spacing
        words = []
        for i, token in enumerate(tokens):
            if i > 0 and not token.word.startswith(("'", "-", ".", ",", "!", "?")):
                words.append(" ")
            words.append(token.word)

        combined_word = "".join(words)

        # Combine and propagate features
        combined_info = {}
        for token in tokens:
            for key, value in token.additional_info.items():
                if key not in combined_info:
                    combined_info[key] = value
                elif (
                    key in ["Gender", "Number", "Person"]
                    and combined_info[key] != value
                ):
                    # For agreement conflicts, prefer leftmost value
                    logger.warning(
                        f"Agreement conflict for {key}: {combined_info[key]} vs {value}"
                    )

        # Create new node
        new_node = ParsedNode(
            word=combined_word,
            lemma=" ".join(token.lemma for token in tokens),
            pos=rule.result,
            weight=sum(token.weight for token in tokens) / len(tokens),
            additional_info=combined_info,
            start=tokens[0].start,
            end=tokens[-1].end,
            children=tokens,
        )

        # Propagate agreement features
        new_node.propagate_agreement()

        return new_node

    def apply_rules(
        self, tokens: List[ParsedNode]
    ) -> Tuple[Optional[ParsedNode], Dict[int, str]]:
        """Enhanced rule application with better handling of complex structures"""
        logger.info("Starting rule application")

        # Pre-process tokens for negative constructions
        tokens = self._preprocess_negative_particles(tokens)

        while len(tokens) > 1:
            original_length = len(tokens)
            changes_made = False
            new_tokens = []
            i = 0

            # Try applying rules at each position
            while i < len(tokens):
                rule_applied = False

                # Try rules of different lengths
                max_pattern_length = min(4, len(tokens) - i)  # Limit pattern length
                for pattern_length in range(2, max_pattern_length + 1):
                    if i + pattern_length > len(tokens):
                        continue

                    candidate_tokens = tokens[i : i + pattern_length]

                    # Try each rule
                    for rule in self.rules:
                        if len(rule.pattern) == pattern_length and self._match_rule(
                            candidate_tokens, rule.pattern
                        ):
                            new_node = self._apply_rule(candidate_tokens, rule)
                            new_tokens.append(new_node)
                            i += pattern_length
                            changes_made = True
                            rule_applied = True
                            break

                    if rule_applied:
                        break

                if not rule_applied:
                    new_tokens.append(tokens[i])
                    i += 1

            tokens = new_tokens

            # If no changes were made and we still have multiple tokens
            if not changes_made or len(tokens) == original_length:
                break

        # Final validation
        if len(tokens) == 1 and tokens[0].pos == Pos.P:
            return tokens[0], {}
        else:
            # Generate helpful feedback
            feedback = {}
            for i, token in enumerate(tokens):
                if token.pos not in [Pos.P, Pos.PUNCT]:
                    feedback[i] = f"unresolved_{token.pos.name.lower()}"

            return None, feedback

    def parse_sentence(self, disambiguator, tokens) -> Optional[ParsedNode]:
        """Parse sentence with improved error recovery"""
        logger.info("Starting sentence parsing")
        max_iterations = 3
        final_feedback = {}

        for i in range(max_iterations):
            if i == 0:
                disambiguated_tokens = disambiguator.initial_disambiguation()
            else:
                disambiguated_tokens = disambiguator.disambiguate_with_feedback(
                    final_feedback
                )

            parsed_tokens = [
                ParsedNode(t.word, t.lemma, t.pos, t.weight, t.additional_info, i, i)
                for i, t in enumerate(disambiguated_tokens)
            ]

            result, feedback = self.apply_rules(parsed_tokens)
            if result is not None:
                return result

            final_feedback = feedback

        raise ParsingError(final_feedback)


def print_parse_tree(node: ParsedNode, indent: str = "", is_last: bool = True):
    """Enhanced parse tree printing with feature display"""
    if node is None:
        print("Failed to parse sentence")
        return

    marker = "└── " if is_last else "├── "
    print(f"{indent}{marker}{node.word} ({node.pos.name})")

    # Print features
    if node.additional_info:
        feature_indent = indent + ("    " if is_last else "│   ") + "    "
        print(f"{feature_indent}Features: {node.additional_info}")

    if node.children:
        for i, child in enumerate(node.children):
            is_last_child = i == len(node.children) - 1
            new_indent = indent + ("    " if is_last else "│   ")
            print_parse_tree(child, new_indent, is_last_child)


if __name__ == "__main__":
    from dataclasses import dataclass

    @dataclass
    class DisambiguatedToken:
        word: str
        lemma: str
        pos: Pos
        additional_info: Dict[str, str]
        weight: float

    # Mock Disambiguator for testing
    class MockDisambiguator:
        def initial_disambiguation(self):
            return [
                DisambiguatedToken(
                    "Le",
                    "le",
                    Pos.DET,
                    {"Type": "Def", "Gender": "Mas", "Number": "Sing"},
                    1.0,
                ),
                DisambiguatedToken(
                    "petit", "petit", Pos.ADJ, {"Gender": "Mas", "Number": "Sing"}, 0.8
                ),
                DisambiguatedToken(
                    "chat", "chat", Pos.NOM, {"Gender": "Mas", "Number": "Sing"}, 1.0
                ),
                DisambiguatedToken(
                    "noir", "noir", Pos.ADJ, {"Gender": "Mas", "Number": "Sing"}, 0.9
                ),
                DisambiguatedToken(
                    "dort",
                    "dormir",
                    Pos.VER,
                    {"Tense": "Pres", "Person": "3", "Number": "Sing"},
                    1.0,
                ),
            ]

        def disambiguate_with_feedback(self, feedback):
            return self.initial_disambiguation()

    def print_parse_tree(node: ParsedNode, indent: str = "", is_last: bool = True):
        """Recursively print the parse tree in a hierarchical format"""
        if node is None:
            print("Failed to parse sentence")
            return

        marker = "└── " if is_last else "├── "
        print(f"{indent}{marker}{node.word} ({node.pos.name})")

        if node.additional_info:
            info_indent = indent + ("    " if is_last else "│   ")
            print(f"{info_indent}    Features: {node.additional_info}")

        if node.children:
            for i, child in enumerate(node.children):
                is_last_child = i == len(node.children) - 1
                new_indent = indent + ("    " if is_last else "│   ")
                print_parse_tree(child, new_indent, is_last_child)

    # Create RuleSystem and MockDisambiguator
    rules = [
        Rule("Adjective + Noun", [Pos.ADJ, Pos.NOM], Pos.GN),
        Rule("Noun + Adjective", [Pos.NOM, Pos.ADJ], Pos.GN),
        Rule("Determiner + GN", [Pos.DET, Pos.GN], Pos.GNDet),
        Rule("GNDet + Adjective", [Pos.GNDet, Pos.ADJ], Pos.GNDet),
        Rule("GNDet + VER", [Pos.GNDet, Pos.VER], Pos.P),
    ]

    rule_system = RuleSystem(rules)
    mock_disambiguator = MockDisambiguator()

    print("\nTesting French sentence parsing:")
    print("================================")

    # Parse the sentence
    try:
        parsed_result = rule_system.parse_sentence(mock_disambiguator, [])

        print("\nParsed sentence structure:")
        print("=========================")
        print_parse_tree(parsed_result)

        print("\nDetailed node information:")
        print("=========================")

        def print_node_details(node: ParsedNode, level: int = 0):
            indent = "  " * level
            print(f"{indent}Node: {node.word}")
            print(f"{indent}- POS: {node.pos.name}")
            print(f"{indent}- Lemma: {node.lemma}")
            print(f"{indent}- Weight: {node.weight:.2f}")
            print(f"{indent}- Span: ({node.start}, {node.end})")
            if node.additional_info:
                print(f"{indent}- Features: {node.additional_info}")
            if node.children:
                print(f"{indent}Children:")
                for child in node.children:
                    print_node_details(child, level + 1)

        print_node_details(parsed_result)

    except ParsingError as e:
        print("Failed to parse sentence.")
        print("Feedback:", e.feedback)
