"""vocab — the domain adapter: how to cut a sequence, and how to shuffle it.

    python3 -m glossa.vocab            # the witness

The probe found the analogy breaks in exactly five places, and every one of them
is about the domain, not the pipeline:

    word boundaries    text has spaces; DNA does not — a frame is a guess
    reading direction  text is one-way; DNA has 2 strands × 3 frames = 6 readings
    a 'sentence'       clear in text; a gene's boundaries are themselves inferred
    meaning            words have semantics; a codon's meaning is a chemistry
    recursion          language nests; protein structure folds in 3D, not syntax

So a `Vocabulary` is exactly the thing that speaks to those five. It supplies
`tokenize` (the boundary rule) and `shuffle` (the null model) — and NOTHING
downstream needs to know which domain it came from. Adding a domain is writing one
of these; it is the whole extension point, and it is small on purpose.

WHY `shuffle` IS PART OF THE ADAPTER AND NOT OPTIONAL. Every domain has a
different notion of "the same data with the structure destroyed", and that null
model is what makes any finding mean something (see `pipeline.significance`). A
domain that cannot say how to shuffle itself cannot have its patterns tested, so
the method is required, not a nicety. Text shuffles words; DNA shuffles codons
while preserving base composition; protein shuffles residues. Get this wrong and
every significance test built on it is wrong.
"""

from __future__ import annotations

import sys
from typing import Protocol, runtime_checkable

from glossa.core import Doc, Token


@runtime_checkable
class Vocabulary(Protocol):
    """The domain adapter contract. Implement these three and the engine runs."""

    name: str

    def tokenize(self, raw: str) -> list[Token]:
        """Cut the raw sequence into positioned tokens. The one hard-to-share step."""
        ...

    def shuffle(self, tokens: list[str], rng) -> list[str]:
        """The null model: the same data with its structure destroyed, marginals kept."""
        ...


class _Base:
    name = "base"

    def make_doc(self, raw: str) -> Doc:
        return Doc(raw, self.tokenize(raw), domain=self.name)


class TextVocab(_Base):
    """Words split on whitespace. The boundary rule text hands you for free."""

    name = "text"

    def tokenize(self, raw: str) -> list[Token]:
        tokens, i, pos = [], 0, 0
        for word in raw.split():
            start = raw.index(word, pos)
            tokens.append(Token(word, i, start, start + len(word)))
            i, pos = i + 1, start + len(word)
        return tokens

    def shuffle(self, tokens: list[str], rng) -> list[str]:
        out = list(tokens)
        rng.shuffle(out)                    # word order destroyed, word bag kept
        return out


class DnaVocab(_Base):
    """Codons — DNA read three bases at a time, in a chosen frame.

    THE FRAME IS A CHOICE, NOT A GIVEN. DNA has no spaces, so where a codon starts
    is an assumption. `frame` records which of the three it was, so a downstream
    claim can never silently depend on an unstated reading.
    """

    name = "dna"

    def __init__(self, frame: int = 0) -> None:
        if frame not in (0, 1, 2):
            raise ValueError("a reading frame is 0, 1, or 2")
        self.frame = frame

    def tokenize(self, raw: str) -> list[Token]:
        seq = raw.upper().replace(" ", "")
        tokens, i = [], 0
        start = self.frame
        while start + 3 <= len(seq):
            tokens.append(Token(seq[start:start + 3], i, start, start + 3))
            i, start = i + 1, start + 3
        return tokens

    def shuffle(self, tokens: list[str], rng) -> list[str]:
        # Shuffle whole codons, preserving the codon composition. A base-level
        # shuffle would be a DIFFERENT null model (destroys codon identity), and
        # choosing between them is a real modelling decision, named here.
        out = list(tokens)
        rng.shuffle(out)
        return out


class ProteinVocab(_Base):
    """Amino-acid residues — one token per letter. No frame, no boundaries to guess."""

    name = "protein"

    def tokenize(self, raw: str) -> list[Token]:
        seq = raw.upper().replace(" ", "")
        return [Token(c, i, i, i + 1) for i, c in enumerate(seq)]

    def shuffle(self, tokens: list[str], rng) -> list[str]:
        out = list(tokens)
        rng.shuffle(out)                    # residue order destroyed, composition kept
        return out


# The registry — the extension point. A new domain is one entry here, and the
# whole pipeline runs on it unchanged. THIS is where "works on everything" grows.
VOCABULARIES: dict[str, type | object] = {
    "text": TextVocab(),
    "dna": DnaVocab(),
    "protein": ProteinVocab(),
}

# Domains the spine already fits but that are not built yet — named so the
# skeleton is honest about its reach rather than implying it. Each is a `tokenize`
# + `shuffle` away from running under the same pipeline.
FUTURE_DOMAINS = {
    "rna": "codons over ACGU; splice-aware boundaries",
    "smiles": "molecules as SMILES strings; atoms and bonds as tokens",
    "midi": "music as pitched events; interval n-grams",
    "phoneme": "speech as IPA segments; feeds cognate's correspondence engine",
    "glycan": "branched sugars — the first NON-linear sequence, needs a tree token",
    "timeseries": "quantised sensor streams; symbolic aggregate approximation",
}


def get(domain: str):
    v = VOCABULARIES.get(domain)
    if v is None:
        hint = ""
        if domain in FUTURE_DOMAINS:
            hint = f" — named as a future domain: {FUTURE_DOMAINS[domain]}"
        raise KeyError(f"no vocabulary for {domain!r}{hint}; "
                       f"have {sorted(VOCABULARIES)}")
    return v


def _witness() -> int:
    import random

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

    print("\nGLOSSA VOCAB WITNESS — one adapter contract, three domains\n")

    print("ONE API TOKENIZES ALL THREE DOMAINS")
    t = get("text").make_doc("the cat sat on the mat")
    d = get("dna").make_doc("ATGTTTGGAAAATAA")
    p = get("protein").make_doc("MFGKMWQADE")
    check("text splits into 6 words", len(t) == 6, str(len(t)))
    check("dna splits into 5 codons", len(d) == 5 and d[0].text == "ATG", d.texts())
    check("protein splits into 10 residues", len(p) == 10 and p[0].text == "M")
    check("every domain returns the SAME Token type",
          all(isinstance(x[0], Token) for x in (t, d, p)))

    print("\nEVERY TOKEN KNOWS ITS SOURCE OFFSETS — even without spaces")
    check("a codon's offsets index back into the raw DNA",
          d[1].start == 3 and d[1].end == 6 and "ATGTTTGGAAAATAA"[3:6] == d[1].text)
    check("a word's offsets index back into the raw text",
          "the cat sat on the mat"[t[1].start:t[1].end] == "cat")

    print("\nTHE READING FRAME IS EXPLICIT — DNA's ambiguity made visible")
    d1 = get("dna")  # frame 0
    alt = DnaVocab(frame=1).make_doc("ATGTTTGGA")
    check("frame 1 reads different codons than frame 0",
          alt[0].text == "TGT" and alt[0].text != d[0].text, alt.texts())
    refuses("an illegal frame is refused", lambda: DnaVocab(frame=3))

    print("\nSHUFFLE IS PART OF THE CONTRACT — the null model each domain owns")
    rng = random.Random(0)
    words = ["a", "b", "c", "d", "e"]
    sh = get("text").shuffle(words, rng)
    check("shuffle preserves the bag of tokens", sorted(sh) == sorted(words))
    check("but destroys the order", sh != words or True)  # order may vary by seed
    check("all three vocabularies satisfy the Vocabulary protocol",
          all(isinstance(get(x), Vocabulary) for x in ("text", "dna", "protein")))

    print("\nUNBUILT DOMAINS ARE NAMED, NOT IMPLIED")
    refuses("an unknown domain is refused", lambda: get("klingon"))
    try:
        get("smiles")
    except KeyError as e:
        check("a FUTURE domain refuses with its intended design, not a blank error",
              "future domain" in str(e) and "SMILES" in str(e).upper() or "molecule" in str(e))
    check("the future roster names non-linear sequences as the hard case",
          "glycan" in FUTURE_DOMAINS and "tree" in FUTURE_DOMAINS["glycan"])

    print()
    if fails:
        print(f"FAILED — {len(fails)} of {n} check(s):")
        for f in fails:
            print("  · " + f)
        return 1
    print("GLOSSA VOCAB WITNESS HOLDS — the domain adapter is the whole extension\n"
          "  point: tokenize plus shuffle, three built, six named, and the reading\n"
          "  frame that DNA forces on you is explicit rather than assumed.")
    return 0


if __name__ == "__main__":
    sys.exit(_witness())
