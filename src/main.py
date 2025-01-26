# main.py

import logging
from typing import List, Optional
from Fetcher import Fetcher
from Loader import WebLoader
from Tokenizer import Tokenizer
from Disambiguator import Disambiguator
from RuleSystem import RuleSystem, ParsedNode, Rule, ParsingError, print_parse_tree
from Token import Pos

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class FrenchNLPPipeline:
    def __init__(self, jdm_url: str = "http://www.jeuxdemots.org/rezo-dump.php"):
        """Initialize the French NLP pipeline with components."""
        logger.info("Initializing French NLP pipeline...")
        self.fetcher = Fetcher(jdm_url)
        self.web_loader = WebLoader(jdm_url)
        self.rule_system = RuleSystem()
        logger.info("Pipeline initialized successfully")

    def process_sentence(self, sentence: str) -> Optional[ParsedNode]:
        """Process a single French sentence through the entire pipeline."""
        try:
            logger.info(f"Processing sentence: {sentence}")

            # Step 1: Tokenization
            logger.info("Tokenizing sentence...")
            tokens = Tokenizer(sentence, self.web_loader, None)
            logger.info(f"Tokens: {[t.word for t in tokens]}")
            logger.info("Individual token analysis:")
            for token in tokens:
                logger.info(f"\nToken: {token.word}")
                logger.info(f"All POS tags: {token.all_pos_tags}")
                logger.info(f"Word POS info: {token._word_pos_info}")
                logger.info(f"Lemma info: {token._lemma_info}")

            disambiguator = Disambiguator(tokens)
            disambiguated = disambiguator.initial_disambiguation()
            logger.info(
                f"Disambiguated tokens: {[(t.word, t.pos.name) for t in disambiguated]}"
            )

            # Step 3: Parse using rule system
            logger.info("Parsing sentence structure...")
            try:
                parsed_result = self.rule_system.parse_sentence(disambiguator, tokens)
                logger.info("Parsing complete")
                return parsed_result
            except ParsingError as e:
                logger.warning(f"Parsing failed: {str(e)}")
                raise

        except Exception as e:
            logger.error(f"Error processing sentence: {str(e)}")
            return None


def run_tests():
    """Run tests with various French sentences."""
    pipeline = FrenchNLPPipeline()

    test_sentences = [
        # Simple sentences
        "Le chat dort.",
        "La petite fille joue.",
        "Je mange une pomme.",
        # Sentences with adjectives
        "Le grand chien noir aboie.",
        "Une belle maison bleue.",
        "Les petits oiseaux chantent.",
        # Sentences with prepositions
        "Le livre est sur la table.",
        "Je vais à Paris.",
        "Le chat dort dans le jardin.",
        # Negative sentences
        "Je ne mange pas.",
        "Il n'aime pas le café.",
        # "Nous ne dormons plus.",
    ]

    print("\nFrench NLP Pipeline Test Results")
    print("================================\n")

    for i, sentence in enumerate(test_sentences, 1):
        print(f'\nTest {i}: "{sentence}"')
        print("-" * (len(sentence) + 15))

        try:
            result = pipeline.process_sentence(sentence)

            if result:
                print("\nParse Tree:")
                print_parse_tree(result)
                print("\nDetailed Analysis:")
                print_detailed_analysis(result)
            else:
                print("Failed to parse sentence")

        except ParsingError as e:
            print(f"Parsing error: {str(e)}")
        except Exception as e:
            print(f"Error: {str(e)}")

        print("\n" + "=" * 80)


def print_detailed_analysis(node: ParsedNode, level: int = 0):
    """Print detailed grammatical analysis of the parsed sentence."""
    indent = "  " * level

    # Print node information
    print(f"{indent}Node: {node.word}")
    print(f"{indent}POS: {node.pos.name}")

    # Print features
    if node.additional_info:
        print(f"{indent}Features:")
        for key, value in node.additional_info.items():
            print(f"{indent}  - {key}: {value}")

    # Print span information
    print(f"{indent}Span: ({node.start}, {node.end})")

    # Recursively print children
    if node.children:
        print(f"{indent}Components:")
        for child in node.children:
            print_detailed_analysis(child, level + 1)


def main():
    """Main function to run the French NLP pipeline tests."""
    try:
        logger.info("Starting French NLP pipeline tests")
        run_tests()
        logger.info("Tests completed successfully")

    except Exception as e:
        logger.error(f"Error during testing: {str(e)}")
        raise


if __name__ == "__main__":
    main()
