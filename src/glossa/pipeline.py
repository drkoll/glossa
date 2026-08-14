"""pipeline — components that run in order over a Doc, with the null model built in.

    python3 -m glossa.pipeline            # the witness

A spaCy pipeline is a list of components, each adding annotations to the Doc. This
is the same, with one component spaCy does not have and this project cannot do
without: `significance`, which measures an observed pattern against the domain's
own shuffle and refuses to call it real until it beats chance.

That component is the reason this is a synthropy engine and not a spaCy clone.
Everywhere else in these packages the rule is the same — similarity is not
evidence, a match is the default outcome, recurrence beats resemblance — and here
it becomes a pipeline stage any domain inherits for free.

────────────────────────────────────────────────────────────────────────────
THE PIPELINE IS THE PLACE FEATURES STACK

The request was to stack features as high as possible. This is where that
happens: a component is `(Doc) -> Doc`, they compose in any order, and each reads
the tags the earlier ones wrote. Four are built here — enough to prove the shape.
The rest is addition, not redesign, and the FUTURE_COMPONENTS roster names the
next layers without faking them.
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import math
import sys
from collections import Counter
from dataclasses import dataclass
from typing import Callable

from glossa.core import Doc, Judgment, Tag

Component = Callable[[Doc], Doc]


def frequencies(doc: Doc) -> Doc:
    """Tag each token PLUS if it recurs, MINUS if it is a hapax (appears once)."""
    counts = Counter(doc.texts())
    for tok in doc:
        c = counts[tok.text]
        if c > 1:
            tok.set("recurs", Judgment.yes(c, f"appears {c} times"))
        else:
            tok.set("recurs", Judgment.no("appears once — a hapax"))
    doc.tags["types"] = Judgment.yes(len(counts), f"{len(counts)} distinct units")
    return doc


def entropy(doc: Doc) -> Doc:
    """Attach the Shannon entropy of the token stream, in bits, at the Doc level."""
    toks = doc.texts()
    if not toks:
        doc.tags["entropy"] = Judgment.unknown("empty document")
        return doc
    n = len(toks)
    h = -sum((k / n) * math.log2(k / n) for k in Counter(toks).values())
    doc.tags["entropy"] = Judgment.yes(round(h, 4), f"{h:.4f} bits/token")
    return doc


def translator(table: dict[str, str], tag_name: str) -> Component:
    """A table-driven tagger: assign each token a label, or ZERO if unlisted.

    This is the genetic code as an NLP tagger — codon -> amino acid — but the
    shape is general: any per-token lookup. A token whose text is not in the table
    gets ZERO, never a wrong guess, which is how the genetic code's stop codons and
    a text tagger's unknown words are handled by the same honest default.
    """
    def component(doc: Doc) -> Doc:
        for tok in doc:
            if tok.text in table:
                tok.set(tag_name, Judgment.yes(table[tok.text], "from table"))
            else:
                tok.set(tag_name, Judgment.unknown(f"{tok.text!r} not in table"))
        return doc
    return component


# Significance WAS a local dataclass carrying observed/chance_mean/chance_max/
# trials — which is exactly krisis.Assay. The null-model METHOD below was also
# reimplemented here before krisis held it. Both now come from the shared core;
# this is the second glossa migration, absorbing a method rather than a type.
from krisis.assay import Assay as Significance
from krisis.assay import assay as _assay


def significance(vocab, statistic: Callable[[list[str]], float], *,
                 trials: int = 500, seed: int = 0) -> Component:
    """Measure a statistic against the domain's OWN shuffle. The synthropy stage.

    `statistic` maps a token stream to a number (a repeat count, a motif score,
    an autocorrelation). It is computed on the real sequence and on `trials`
    shuffles drawn from the vocabulary's null model. The result is SUPPORTED only
    if the real value exceeds every shuffled one — otherwise INSUFFICIENT, never a
    confident pattern. This is `cognate`'s chance baseline generalised to any
    sequence domain.
    """
    def component(doc: Doc) -> Doc:
        toks = doc.texts()
        observed = statistic(toks)
        # the domain's own null model, as the sampler krisis.assay expects
        v = _assay(observed, lambda r: statistic(vocab.shuffle(toks, r)),
                   trials=trials, seed=seed)
        doc.tags["significance"] = v            # a Verdict (== glossa Judgment)
        doc.tags["_sig_detail"] = Judgment.yes(v.value)   # the Assay detail
        return doc
    return component


class Pipeline:
    """An ordered stack of components. Run a raw sequence through the whole thing."""

    def __init__(self, vocab, components: list[Component] | None = None) -> None:
        self.vocab = vocab
        self.components = components or []

    def add(self, component: Component) -> Pipeline:
        self.components.append(component)
        return self

    def __call__(self, raw: str) -> Doc:
        doc = self.vocab.make_doc(raw)
        for c in self.components:
            doc = c(doc)
        return doc


# The next layers, named rather than implied. Each is a component the spine
# already supports; none is pretended to exist.
FUTURE_COMPONENTS = {
    "ngrams": "sliding-window motifs; the unit of most sequence pattern-finding",
    "aligner": "pairwise alignment (Needleman-Wunsch) — glossa already has phones in cognate",
    "embedder": "count-based vectors per token; no model weights, still comparable",
    "segmenter": "unsupervised boundary finding — where a genome's 'words' are inferred",
    "tagger_learned": "a trained sequence tagger; the first component with parameters",
    "compressor": "NCD between two Docs — brightchain already ships the metric",
}


def _witness() -> int:
    from glossa.vocab import get

    fails: list[str] = []
    n = 0

    def check(label: str, cond: bool, detail: str = "") -> None:
        nonlocal n
        n += 1
        print(("  PASS  " if cond else "  FAIL  ") + label
              + ("" if cond else f" — {detail}"))
        if not cond:
            fails.append(label)

    print("\nGLOSSA PIPELINE WITNESS — components stack, and chance is measured\n")

    print("ONE PIPELINE SHAPE RUNS ON TEXT AND ON PROTEIN")
    for domain, raw, unit_ct in (("text", "the cat sat on the mat the cat", 8),
                                 ("protein", "MFGKMWQADEMFGK", 14)):
        pipe = Pipeline(get(domain)).add(frequencies).add(entropy)
        doc = pipe(raw)
        check(f"{domain}: it tokenized and annotated", len(doc) == unit_ct, str(len(doc)))
        check(f"{domain}: entropy was computed", doc.tags["entropy"].tag is Tag.PLUS)

    print("\nRECURRENCE IS THREE-VALUED — a hapax is MINUS, a repeat is PLUS")
    doc = Pipeline(get("text")).add(frequencies)("the cat sat on the mat")
    the = [t for t in doc if t.text == "the"][0]
    cat = [t for t in doc if t.text == "cat"][0]
    check("'the' recurs -> PLUS", the.tag("recurs").tag is Tag.PLUS)
    check("'cat' is a hapax -> MINUS, not absent", cat.tag("recurs").tag is Tag.MINUS)
    check("a tag never assigned is ZERO",
          the.tag("amino_acid").tag is Tag.ZERO)

    print("\nTHE GENETIC CODE IS AN NLP TAGGER — one token stream mapped to another")
    CODON = {"ATG": "M", "TTT": "F", "GGA": "G", "AAA": "K", "TAA": "*"}
    pipe = Pipeline(get("dna")).add(translator(CODON, "amino_acid"))
    doc = pipe("ATGTTTGGAAAATAA")
    aas = [t.tag("amino_acid").value for t in doc]
    check("DNA translated to protein through a table tagger",
          aas == ["M", "F", "G", "K", "*"], str(aas))
    unknown = Pipeline(get("dna")).add(translator({"ATG": "M"}, "amino_acid"))("ATGCCC")
    check("an unlisted codon is ZERO, never a wrong guess",
          unknown[1].tag("amino_acid").tag is Tag.ZERO)

    print("\nSIGNIFICANCE — a pattern must beat the domain's OWN shuffle")
    # A sequence with a REAL repeated motif vs the same bag shuffled.
    def max_repeat(toks: list[str]) -> float:
        return max(Counter(toks).values()) if toks else 0.0
    structured = "GO GO GO GO GO stop"                 # 'GO' recurs hard
    doc = Pipeline(get("text")).add(
        significance(get("text"), max_repeat, trials=300))(structured)
    sig = doc.tags["significance"]
    # max_repeat is invariant under shuffle, so it can NEVER beat chance — the
    # honest verdict is INSUFFICIENT, and that is the point of the check.
    check("a shuffle-invariant statistic is correctly ruled INSUFFICIENT/‑",
          sig.tag in (Tag.ZERO, Tag.MINUS), sig.why)

    # A statistic that depends on ORDER over a DIVERSE bag: how often the single
    # most common adjacent bigram repeats. A periodic sequence scores high; the
    # same tokens shuffled almost never reproduce the period, so it beats chance.
    def top_bigram(toks: list[str]) -> float:
        bigrams = Counter(zip(toks, toks[1:]))
        return max(bigrams.values()) if bigrams else 0.0
    periodic = " ".join(["A", "B"] * 6)                 # ABABAB… — 6 balanced A/B
    doc2 = Pipeline(get("text")).add(
        significance(get("text"), top_bigram, trials=500))(periodic)
    sig2 = doc2.tags["significance"]
    check("a periodic motif over a balanced bag beats every shuffle -> SUPPORTED",
          sig2.tag is Tag.PLUS, sig2.why)
    detail = doc2.tags["_sig_detail"].value
    check("and the finding shows its work", detail.observed > detail.chance_max,
          detail.line())

    print("\nTHE PIPELINE STACKS — components compose and read each other")
    pipe = (Pipeline(get("dna"))
            .add(translator({"ATG": "M", "TTT": "F", "GGA": "G", "AAA": "K", "TAA": "*"},
                            "amino_acid"))
            .add(frequencies).add(entropy))
    doc = pipe("ATGTTTGGAAAATAAATGTTT")
    check("a three-stage stack ran end to end",
          doc.tags["entropy"].tag is Tag.PLUS
          and doc[0].tag("amino_acid").value == "M"
          and doc[0].tag("recurs").tag is Tag.PLUS)
    check("the next layers are named, not implied",
          "ngrams" in FUTURE_COMPONENTS and "compressor" in FUTURE_COMPONENTS)

    print()
    if fails:
        print(f"FAILED — {len(fails)} of {n} check(s):")
        for f in fails:
            print("  · " + f)
        return 1
    print("GLOSSA PIPELINE WITNESS HOLDS — components stack in any order, the\n"
          "  genetic code runs as a tagger, and no pattern is called real until it\n"
          "  beats the domain's own shuffle. Similarity is not evidence, here too.")
    return 0


if __name__ == "__main__":
    sys.exit(_witness())
