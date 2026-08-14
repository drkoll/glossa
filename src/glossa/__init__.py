"""glossa — a ternary NLP engine for any sequence: text, genes, proteins, and more.

    from glossa import Pipeline, get
    from glossa.pipeline import frequencies, entropy, translator, significance

    pipe = Pipeline(get("dna")).add(translator(GENETIC_CODE, "amino_acid"))
    doc = pipe("ATGTTTGGAAAATAA")          # -> M F G K *

γλῶσσα, the tongue. spaCy assumes text; the probe showed the same pipeline runs on
codons and residues once tokenization is factored out. So glossa keeps the token/
pipeline SPINE and pushes every domain difference into a small `Vocabulary`
adapter — and it is three-valued throughout, so "not enough to say" is a real
answer rather than a silent false. The chance baseline is a built-in pipeline
stage: no pattern is called real until it beats the domain's own shuffle.
"""

from __future__ import annotations

from glossa.bounds import (
    BREAKS,
    Break,
    bounded,
    depth_profile,
    nesting,
    parse_nesting,
    readings,
)
from glossa.core import Doc, Judgment, Tag, Token
from glossa.pipeline import (
    FUTURE_COMPONENTS,
    Pipeline,
    Significance,
    entropy,
    frequencies,
    significance,
    translator,
)
from glossa.vocab import (
    FUTURE_DOMAINS,
    VOCABULARIES,
    DnaVocab,
    ProteinVocab,
    TextVocab,
    Vocabulary,
    get,
)

__version__ = "0.1.0"
__all__ = [
    "Doc", "Token", "Tag", "Judgment",
    "Vocabulary", "TextVocab", "DnaVocab", "ProteinVocab", "get",
    "VOCABULARIES", "FUTURE_DOMAINS",
    "Pipeline", "frequencies", "entropy", "translator", "significance",
    "Significance", "FUTURE_COMPONENTS",
    # bounds — the five breaks mathematised, with recursion-breakers
    "BREAKS", "Break", "parse_nesting", "depth_profile", "bounded", "readings",
    "nesting",
    "__version__",
]
