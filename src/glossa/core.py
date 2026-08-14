"""core — the domain-agnostic sequence model: Doc, Token, and a three-valued tag.

    python3 -m glossa.core            # the witness

A spaCy Doc is a sequence of Tokens over text. This is the same shape with the
text assumption removed, because the probe showed the same downstream analysis
runs on words, codons, and residues once tokenization is factored out. So the
core knows about SEQUENCES OF DISCRETE UNITS and nothing about language.

────────────────────────────────────────────────────────────────────────────
WHY THE TAGS ARE THREE-VALUED

spaCy's tags are present or absent. That is a two-valued world, and it hides the
distinction this whole project turns on: a property that was CHECKED AND FALSE is
not the same as one that was NEVER CHECKED. A codon with no known amino acid, a
word with no attested etymology, a residue whose structure was not computed — all
read as "false" in a binary tagger, and all three are lies.

So a `Tag` is PLUS, ZERO, or MINUS. ZERO is not a failure state and never reads as
false — it is the honest answer "not enough to say", and it is the most common
answer an engine spanning genomes and languages will give. A component that cannot
decide writes ZERO rather than guessing, and ZERO propagates rather than
collapsing. This is the Kleene logic the other packages use, made native to the
token model here.
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Iterator


class Tag(IntEnum):
    """A three-valued judgement about a token property. ZERO is not false."""

    PLUS = 1        # the property holds
    ZERO = 0        # not enough to say — the honest middle, never 'false'
    MINUS = -1      # the property was checked and does not hold

    @property
    def glyph(self) -> str:
        return {1: "+", 0: "0", -1: "−"}[int(self)]

    def __repr__(self) -> str:
        return f"Tag.{self.name}"


@dataclass(frozen=True)
class Judgment:
    """A tag with its reason. A finding that cannot say WHY is not a finding."""

    tag: Tag
    why: str = ""
    value: Any = None       # the concrete label, present only when tag is PLUS

    def __bool__(self) -> bool:
        raise TypeError(
            "a Judgment is three-valued and refuses to collapse to two. Test "
            "`j.tag is Tag.PLUS` — an implicit bool would read ZERO (not enough "
            "to say) as False, which is the defect this type prevents.")

    @classmethod
    def yes(cls, value: Any = None, why: str = "") -> Judgment:
        return cls(Tag.PLUS, why, value)

    @classmethod
    def no(cls, why: str = "") -> Judgment:
        return cls(Tag.MINUS, why)

    @classmethod
    def unknown(cls, why: str = "") -> Judgment:
        return cls(Tag.ZERO, why)


@dataclass
class Token:
    """One discrete unit in a sequence. A word, a codon, a residue — the same shape.

    `text` is the surface form. `i` is its position. `start`/`end` are offsets into
    the raw sequence, so a token always knows where it came from — the audit trail
    that lets a downstream claim be traced back to the source bytes.
    """

    text: str
    i: int
    start: int
    end: int
    tags: dict[str, Judgment] = field(default_factory=dict)

    def tag(self, name: str) -> Judgment:
        """A tag never set is ZERO — unread, not false. The core's central rule."""
        return self.tags.get(name, Judgment.unknown(f"'{name}' was never assigned"))

    def set(self, name: str, judgment: Judgment) -> None:
        self.tags[name] = judgment

    def __repr__(self) -> str:
        marks = "".join(f" {k}={j.tag.glyph}" for k, j in self.tags.items())
        return f"<{self.text!r}@{self.i}{marks}>"


class Doc:
    """A tokenized sequence over one domain. Iterable and sliceable, like a Doc.

    The Doc does not tokenize — a `Vocabulary` does, because HOW to cut a sequence
    into units is the one thing that genuinely differs across domains (text has
    spaces; DNA does not). The Doc holds the result and the domain it came from.
    """

    def __init__(self, raw: str, tokens: list[Token], *, domain: str) -> None:
        self.raw = raw
        self._tokens = tokens
        self.domain = domain
        self.tags: dict[str, Judgment] = {}          # document-level judgements

    def __len__(self) -> int:
        return len(self._tokens)

    def __iter__(self) -> Iterator[Token]:
        return iter(self._tokens)

    def __getitem__(self, i):
        return self._tokens[i]              # int -> Token, slice -> list[Token]

    def texts(self) -> list[str]:
        return [t.text for t in self._tokens]

    def with_tag(self, name: str, tag: Tag) -> list[Token]:
        return [t for t in self._tokens if t.tag(name).tag is tag]

    def __repr__(self) -> str:
        return f"<Doc {self.domain}: {len(self)} tokens>"


def _witness() -> int:
    fails: list[str] = []
    n = 0

    def check(label: str, cond: bool, detail: str = "") -> None:
        nonlocal n
        n += 1
        print(("  PASS  " if cond else "  FAIL  ") + label
              + ("" if cond else f" — {detail}"))
        if not cond:
            fails.append(label)

    def refuses(label: str, fn) -> None:
        try:
            fn()
            check(label, False, "the operation was ALLOWED")
        except Exception:
            check(label, True)

    print("\nGLOSSA CORE WITNESS — the domain-agnostic sequence model\n")

    print("A TAG IS THREE-VALUED, AND ZERO IS NOT FALSE")
    check("PLUS, ZERO, MINUS exist and ZERO sits between", Tag.MINUS < Tag.ZERO < Tag.PLUS)
    check("their glyphs render", (Tag.PLUS.glyph, Tag.ZERO.glyph, Tag.MINUS.glyph)
          == ("+", "0", "−"))
    refuses("a Judgment refuses to collapse to bool",
            lambda: bool(Judgment.unknown("test")))
    check("an unknown judgment is ZERO, not MINUS",
          Judgment.unknown().tag is Tag.ZERO)
    check("a 'no' is MINUS — checked and false, distinct from unknown",
          Judgment.no("checked").tag is Tag.MINUS)

    print("\nA TOKEN THAT WAS NEVER TAGGED READS ZERO — the core's central rule")
    tok = Token("ATG", 0, 0, 3)
    check("an unset tag is ZERO, not absent-as-false",
          tok.tag("amino_acid").tag is Tag.ZERO)
    check("and it says why", "never assigned" in tok.tag("amino_acid").why)
    tok.set("amino_acid", Judgment.yes("M", "start codon"))
    check("once set, it is PLUS with a value", tok.tag("amino_acid").tag is Tag.PLUS
          and tok.tag("amino_acid").value == "M")

    print("\nA DOC IS A SLICEABLE SEQUENCE THAT KNOWS ITS DOMAIN")
    toks = [Token(c, i, i, i + 1) for i, c in enumerate("MFGK")]
    doc = Doc("MFGK", toks, domain="protein")
    check("it has a length", len(doc) == 4)
    check("it is indexable", doc[0].text == "M")
    check("it is sliceable into tokens", [t.text for t in doc[1:3]] == ["F", "G"])
    check("it reports its domain", doc.domain == "protein")
    check("it can filter by three-valued tag",
          doc.with_tag("amino_acid", Tag.PLUS) == [])
    check("every token carries its source offsets for audit",
          all(t.end - t.start == 1 for t in doc))

    print()
    if fails:
        print(f"FAILED — {len(fails)} of {n} check(s):")
        for f in fails:
            print("  · " + f)
        return 1
    print("GLOSSA CORE WITNESS HOLDS — one sequence model over any domain, tags\n"
          "  that are three-valued so 'unread' never reads as 'false', and every\n"
          "  token tracing back to the bytes it came from.")
    return 0


if __name__ == "__main__":
    sys.exit(_witness())
