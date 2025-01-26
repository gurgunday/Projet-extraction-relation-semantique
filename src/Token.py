# Token.py

from Loader import WebLoader
from enum import Enum, auto
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass


class Pos(Enum):
    NOM = auto()
    DET = auto()
    ADJ = auto()
    ADV = auto()
    VER = auto()
    PRO = auto()
    PRE = auto()
    CONJ = auto()
    PUNCT = auto()
    NEG = auto()

    P = auto()
    GN = auto()
    GNDet = auto()
    GV = auto()
    GVNeg = auto()
    GP = auto()
    GAdj = auto()
    GAdv = auto()
    RelativeClause = auto()

    UNK = auto()


@dataclass
class PosInfo:
    pos: str  # Store the raw POS string
    weight: float
    additional_info: List[str]


@dataclass
class LemmaInfo:
    lemma: str
    weight: float
    pos_info: List[PosInfo]


class Token:
    LEMMA_RELATION_ID = "19"
    POS_RELATION_ID = "4"

    def __init__(self, word: str, web_loader: WebLoader):
        self.web_loader = web_loader
        self.word = word
        self._word_pos_info: List[PosInfo] = []
        self._lemma_info: List[LemmaInfo] = []
        self._analyze()

    def _analyze(self) -> None:
        word_pos_data = self.web_loader.get_entry(
            f"{self.word}:{self.POS_RELATION_ID}:"
        )
        word_pos_data_entries = word_pos_data.get("e", [])

        if len(word_pos_data_entries) > 1:
            for entry in word_pos_data_entries[1:]:
                infos = entry[1].split(":")  # e.g. 'Ver:IPre+SG+P1:IPre+SG+P3'
                pos = infos[0]
                additional_info = infos[1:] if len(infos) > 1 else []

                self._word_pos_info.append(
                    PosInfo(
                        pos=pos, weight=float(entry[3]), additional_info=additional_info
                    )
                )

        # Get lemma information
        lemma_data = self.web_loader.get_entry(f"{self.word}:{self.LEMMA_RELATION_ID}:")
        lemma_data_entries = lemma_data.get("e", [])

        for entry in lemma_data_entries:
            lemma = entry[1]
            lemma_weight = float(entry[3])

            if lemma_weight <= 0.0:
                continue

            # Get POS information for each lemma
            lemma_pos_data = self.web_loader.get_entry(
                f"{lemma}:{self.POS_RELATION_ID}:"
            )
            lemma_pos_info = []
            lemma_pos_data_entries = lemma_pos_data.get("e", [])

            if len(lemma_pos_data_entries) > 1:
                for pos_entry in lemma_pos_data_entries[1:]:
                    infos = pos_entry[1].split(":")
                    pos = infos[0]
                    additional_info = infos[1:] if len(infos) > 1 else []

                    lemma_pos_info.append(
                        PosInfo(
                            pos=pos,
                            weight=float(pos_entry[3]),
                            additional_info=additional_info,
                        )
                    )

            self._lemma_info.append(
                LemmaInfo(lemma=lemma, weight=lemma_weight, pos_info=lemma_pos_info)
            )

    @property
    def lemmas(self) -> List[str]:
        """Get all possible lemmas"""
        return [info.lemma for info in self._lemma_info]

    @property
    def all_pos_tags(self) -> Set[str]:
        """Get all possible POS tags across all lemmas"""
        word_pos = {info.pos for info in self._word_pos_info}
        lemma_pos = {
            pos_info.pos for info in self._lemma_info for pos_info in info.pos_info
        }
        return word_pos.union(lemma_pos)

    def get_pos_for_lemma(self, lemma: str) -> List[PosInfo]:
        """Get possible POS tags for a specific lemma"""
        for info in self._lemma_info:
            if info.lemma == lemma:
                return info.pos_info
        return [PosInfo(pos="UNK", weight=1.0, additional_info=[])]

    def __str__(self) -> str:
        """String representation of the token"""
        parts = []

        word_pos_str = [
            f"{pos_info.pos}({pos_info.weight:.2f})" for pos_info in self._word_pos_info
        ]
        if word_pos_str:
            parts.append(f"Word POS: {', '.join(word_pos_str)}")

        for info in self._lemma_info:
            pos_str = [
                f"{pos_info.pos}({pos_info.weight:.2f})" for pos_info in info.pos_info
            ]
            parts.append(f"{info.lemma}({info.weight:.2f}): {', '.join(pos_str)}")

        return f"Token('{self.word}') -> {'; '.join(parts)}"


if __name__ == "__main__":
    token = Token("aille", WebLoader(""))

    print("Lemmas:", token.lemmas)

    print("All POS tags:", token.all_pos_tags)

    print(f"POS for '{token}':", token.get_pos_for_lemma("ailler"))

    print(token)
