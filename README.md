# glossa

**A ternary NLP engine for any sequence — text, genes, proteins, and whatever
else turns out to be language. spaCy assumes text; glossa doesn't.**

Pure Python standard library. No dependencies, no model weights. A foundational
skeleton, built to grow — not a finished engine.

```bash
pip install glossa
python3 -m glossa            # all three witnesses
```

γλῶσσα — the tongue.

## The probe that set the architecture

A gene, a protein, and a sentence: are they one kind of object a single pipeline
can read? Measured before building anything —

```
domain   tokens  types  entropy
  text        9      6    2.419
   dna        5      5    2.322    (codons)
protein      10      9    3.122    (residues)
```

**The same entropy function ran on all three.** Tokenization differs by one
domain-supplied rule; everything after it is identical. And the genetic code is
just a tagger — `ATG TTT GGA AAA TAA → M F G K *` is one token stream mapped to
another through a table, exactly what a part-of-speech tagger does.

But the analogy breaks in **exactly five places**, and that's the whole design:

| | text | DNA / protein |
|---|---|---|
| word boundaries | spaces give them | none — a reading frame is a *guess* |
| reading direction | one way | 2 strands × 3 frames = 6 readings |
| a "sentence" | clear | a gene's boundaries are themselves inferred |
| meaning | semantics | a codon's meaning is a chemistry |
| recursion | clauses nest | protein folds in 3D, not in syntax |

Every break is about the *domain*, not the *pipeline*. So the shared part becomes
the core, and the five differences are pushed into a small adapter.

## One pipeline, any domain

```python
from glossa import Pipeline, get
from glossa.pipeline import translator, entropy

GENETIC = {"ATG": "M", "TTT": "F", "GGA": "G", "AAA": "K", "TAA": "*"}
doc = Pipeline(get("dna")).add(translator(GENETIC, "aa")).add(entropy)("ATGTTTGGAAAATAA")
# codons -> protein MFGK*, entropy attached — the engine never knew it was DNA
```

Swap `get("dna")` for `get("text")` or `get("protein")` and the same components
run unchanged.

## Three-valued throughout — "unread" is never "false"

spaCy tags are present or absent. That hides the distinction this project turns
on: a property **checked and false** is not a property **never checked**. A codon
with no known amino acid, a word with no attested etymology — both read as
"false" in a binary tagger, and both are lies.

```python
tok.tag("amino_acid")        # Tag.ZERO — "never assigned", not False
```

`Tag` is `PLUS` / `ZERO` / `MINUS`, and `ZERO` never collapses to false. It's the
most common answer an engine spanning genomes and languages will give, and it's
an answer, not a failure. A `Judgment` raises if you try to `bool()` it.

## The one component spaCy doesn't have

`significance` measures an observed pattern against the domain's **own shuffle**,
and refuses to call it real until it beats chance:

```
protein  motif test: +  exceeds all 400 shuffles (max 5.000)     <- SUPPORTED
noise    motif test: −  at or below chance mean 1.000            <- REFUTED
```

This is the discipline every synthropy package shares — similarity is not
evidence, recurrence beats resemblance — made into a pipeline stage any domain
inherits for free. It's why this is a synthropy engine and not a spaCy clone.

Because the null model differs by domain (text shuffles words; DNA shuffles codons
keeping base composition; protein shuffles residues), **`shuffle` is a required
part of the domain adapter.** A domain that can't say how to destroy its own
structure can't have its patterns tested.

## The extension points — where this grows

This is a skeleton. The two registries are the seams meant to be filled, and both
name what's coming rather than implying it:

**Domains** (`get()` today: text, dna, protein). Named next:

```
rna         codons over ACGU; splice-aware boundaries
smiles      molecules as SMILES strings; atoms and bonds as tokens
midi        music as pitched events; interval n-grams
phoneme     speech as IPA segments; feeds cognate's correspondence engine
glycan      branched sugars — the first NON-linear sequence, needs a tree token
timeseries  quantised sensor streams; symbolic aggregate approximation
```

**Components** (today: frequencies, entropy, translator, significance). Named next:
n-grams, pairwise aligner, count embedder, unsupervised segmenter, learned tagger,
and an NCD compressor — several of which already exist in sibling packages
(`cognate` has the aligner and phones; `brightchain` ships NCD).

## Honest scope

- **This is a skeleton, deliberately.** Four components and three domains prove
  the shape. It is not a competitive NLP toolkit today, and won't be until the
  registries above are filled. The ground is set; the building is future work.
- **`glycan` is flagged as the hard case for a reason.** Everything here assumes a
  *linear* sequence. Branched structures (sugars, RNA secondary structure, syntax
  trees) need a tree-shaped token the core does not yet have. That's the boundary
  where "everything is a sequence" genuinely stops.
- **No learned parameters yet.** Every component is deterministic and rule-based.
  The first trained tagger is named in the roster but not built — and it will be
  the first piece that needs data, evaluation, and all the discipline `basanos`
  brings to overfitting.
- **"Meaning" is out of scope, on purpose.** glossa finds structure — recurrence,
  motifs, translations — not semantics. A codon's "meaning" is a chemistry and a
  word's is a world; neither is something a shuffle test can reach.
