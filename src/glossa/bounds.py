"""bounds — the five breaks defined mathematically, each with a recursion-breaker.

    python3 -m glossa.bounds            # the witness

The probe measured every place the language analogy breaks as a SEARCH, and the
sizes told the whole story:

    break              search size          what it is
    word boundaries    2^(n-1)              exponential — segmentation
    reading direction  6                    a FINITE group, Z2 × Z3
    sentence bounds    2^(n-1)              exponential — segmentation again
    meaning            ∞                    needs a world, not a sequence
    recursion/nesting  ∞ (depth)            the only one that self-calls

Four are large-but-finite or bounded. Recursion is the one whose DEPTH is
unbounded — it can call itself forever — so it is the one that needs a breaker
rather than merely a budget. That is why the request named "recursion-breakers":
the deepest break is the one that does not halt on its own.

────────────────────────────────────────────────────────────────────────────
THE ONE IDEA: A RECURSION-BREAKER IS A BOUNDED COMPUTATION WITH A THREE-VALUED
HALT

Every break becomes tractable the same way. Impose a budget; run; and when the
budget is exceeded return ZERO — "not enough to say" — never a crash and never a
guess. This unifies two things that looked different:

    ZERO from missing DATA      the tag was never assigned
    ZERO from missing BUDGET    the search ran out before it could decide

Both are the same honest middle. An engine that must handle "anything we throw at
it" cannot promise to finish, so its promise instead is: it will always halt, and
when it cannot decide within budget it says so in the same three-valued language
it uses for everything else. That is the discernment core — bounded, total, and
never silently false.

WHERE MATH ENDS AND TRAINING BEGINS, STATED PLAINLY. These breakers make the
STRUCTURE of any domain tractable: how deep, how many frames, where the cuts
could be. They do not supply the domain's CONTENT — which cut is right, what a
codon does, what a word means. That is what training data is for. The skeleton is
the discernment; the data is the knowledge. Confusing the two is how a project
promises "handles anything" and ships something that hallucinates.
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Callable

from glossa.core import Doc, Judgment, Tag

__all__ = [
    "BREAKS", "Break", "depth_profile", "parse_nesting", "bounded",
    "readings", "nesting", "BRACKETS",
]

# The canonical bracket pairs. RNA secondary structure in dot-bracket notation and
# a nested clause are the SAME formal object — a Dyck word — so one table reads both.
BRACKETS = {")": "(", "]": "[", "}": "{", ">": "<"}


@dataclass(frozen=True)
class Break:
    """One place the sequence analogy breaks, defined by its search and its breaker."""

    name: str
    search: str             # the size of the naive search, as math
    breaker: str            # the bounding strategy that makes it halt
    status: str             # built | bounded | refused — honest about what exists

    def line(self) -> str:
        return f"{self.name:<18} search {self.search:<12} → {self.breaker} [{self.status}]"


# The mathematical map the request asked for: every break, its search size, and
# the recursion-breaker that bounds it. `status` does not overstate — only nesting
# and reading-direction are BUILT here; segmentation is bounded-in-principle with a
# stub; meaning is REFUSED by design.
BREAKS: tuple[Break, ...] = (
    Break("word_boundaries", "2^(n-1)", "max-segment length L → O(nL); ambiguous ⇒ ZERO", "bounded"),
    Break("reading_direction", "|Z2×Z3|=6", "enumerate the finite group; tie ⇒ ZERO", "built"),
    Break("sentence_bounds", "2^(n-1)", "changepoint in a bounded window", "bounded"),
    Break("meaning", "∞", "REFUSED — a referent is not in the sequence; always ZERO", "refused"),
    Break("recursion", "∞ depth", "bounded-depth pushdown; depth>D ⇒ ZERO", "built"),
)


# ── recursion: the Dyck breaker ─────────────────────────────────────────────
def depth_profile(units: list[str]) -> tuple[int | None, list[int]]:
    """Per-unit nesting depth over a token stream. None if the structure is broken.

    A unit that IS a bracket opens or closes nesting; anything else sits at the
    current depth. Returns (max_depth, profile), or (None, profile) when the
    brackets do not balance — which is a MINUS (a real structural error), distinct
    from merely being too deep.
    """
    depth, prof, stack = 0, [], []
    opens = set(BRACKETS.values())
    for u in units:
        if u in opens:
            depth += 1
            stack.append(u)
            prof.append(depth)
        elif u in BRACKETS:
            if not stack or stack[-1] != BRACKETS[u]:
                return None, prof            # unbalanced or mismatched
            prof.append(depth)
            depth -= 1
            stack.pop()
        else:
            prof.append(depth)
    if stack:
        return None, prof                    # unclosed
    return (max(prof) if prof else 0), prof


def parse_nesting(units: list[str], *, max_depth: int = 32) -> Judgment:
    """THE RECURSION-BREAKER. Three-valued: parsed, too-deep, or malformed.

    PLUS   the structure is balanced and within the depth budget
    ZERO   balanced so far as seen but deeper than the budget — not enough budget
    MINUS  the brackets do not balance — a genuine structural error

    The MINUS/ZERO distinction is the point: an unbalanced structure is WRONG,
    while a too-deep one is merely UNDECIDED at this budget. Collapsing them — the
    thing a two-valued parser must do — throws away exactly the information a
    discernment engine needs.
    """
    if max_depth < 1:
        raise ValueError("a depth budget below 1 can parse nothing")
    d, prof = depth_profile(units)
    if d is None:
        return Judgment.no("brackets do not balance — a structural error")
    if d > max_depth:
        return Judgment.unknown(
            f"depth {d} exceeds the budget of {max_depth} — too deep to say, "
            "the same ZERO as too-little-data")
    return Judgment.yes(d, f"balanced within budget (depth {d} ≤ {max_depth})")


# ── the general breaker: any self-recursive computation, bounded ────────────
def bounded(fn: Callable[..., object], *args, max_steps: int = 10_000) -> Judgment:
    """Run a possibly-non-halting computation under a hard step budget.

    `fn` receives a `tick` callable it MUST call at each recursive step; when the
    budget is spent, `tick` raises and `bounded` returns ZERO rather than looping
    forever. This is the recursion-breaker generalised past brackets: any search
    that might not terminate is made total, and budget exhaustion is ZERO — the
    same honest middle as missing data.
    """
    if max_steps < 1:
        raise ValueError("a computation needs at least one step")
    steps = [0]

    class _Budget(Exception):
        pass

    def tick() -> None:
        steps[0] += 1
        if steps[0] > max_steps:
            raise _Budget

    try:
        result = fn(tick, *args)
    except _Budget:
        return Judgment.unknown(
            f"exceeded the budget of {max_steps} steps — halted at ZERO rather "
            "than running forever")
    except RecursionError:
        return Judgment.unknown(
            "hit the interpreter recursion limit — halted at ZERO")
    return Judgment.yes(result, f"completed in {steps[0]} steps")


# ── reading direction: the finite group Z2 × Z3 ─────────────────────────────
def readings(dna: str) -> dict[str, str]:
    """All six readings of a DNA sequence: 3 frames × 2 strands = the group Z2×Z3.

    Break #2 is not unbounded at all — it is a finite group of order six, so the
    breaker is simply to enumerate it. The point of mathematising it is to show
    the search is CLOSED: there is no seventh reading to worry about, so the engine
    can consider all of them in one bounded pass and pick or return ZERO on a tie.
    """
    seq = dna.upper().replace(" ", "")
    comp = {"A": "T", "T": "A", "G": "C", "C": "G", "N": "N"}
    rc = "".join(comp.get(b, "N") for b in reversed(seq))
    out = {}
    for strand, s in (("+", seq), ("−", rc)):
        for frame in (0, 1, 2):
            out[f"{strand}{frame}"] = s[frame:]
    return out


# ── the pipeline component ──────────────────────────────────────────────────
def nesting(*, max_depth: int = 32) -> Callable[[Doc], Doc]:
    """A pipeline component: tag each token with its nesting depth, the Doc with
    the recursion-breaker's three-valued verdict."""
    def component(doc: Doc) -> Doc:
        units = doc.texts()
        _, prof = depth_profile(units)
        for tok, depth in zip(doc, prof):
            tok.set("depth", Judgment.yes(depth, f"nesting depth {depth}"))
        doc.tags["nesting"] = parse_nesting(units, max_depth=max_depth)
        return doc
    return component


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

    print("\nGLOSSA BOUNDS WITNESS — the five breaks, mathematised and bounded\n")

    print("THE FIVE BREAKS ARE DEFINED, AND THE MAP IS HONEST ABOUT WHAT EXISTS")
    check("all five breaks are catalogued", len(BREAKS) == 5)
    built = {b.name for b in BREAKS if b.status == "built"}
    check("recursion and reading-direction are BUILT",
          built == {"recursion", "reading_direction"}, str(built))
    check("meaning is REFUSED by design, not pretended",
          [b for b in BREAKS if b.name == "meaning"][0].status == "refused")

    print("\nNESTING IS THE DYCK LANGUAGE — one reader for RNA and for syntax")
    rna = list("(((....)))")
    clause = list("((()))")
    check("an RNA hairpin has depth 3", depth_profile(rna)[0] == 3)
    check("a nested clause has depth 3", depth_profile(clause)[0] == 3)
    check("a clover-leaf reads depth 2", depth_profile(list("((..))..((..))"))[0] == 2)
    check("flat sequence has depth 0", depth_profile(list("...."))[0] == 0)

    print("\nTHE RECURSION-BREAKER: three-valued, and MINUS ≠ ZERO")
    deep = list("(" * 8 + ")" * 8)
    check("depth-8 within a budget of 10 → PLUS",
          parse_nesting(deep, max_depth=10).tag is Tag.PLUS)
    over = parse_nesting(deep, max_depth=3)
    check("depth-8 over a budget of 3 → ZERO (too deep, not wrong)",
          over.tag is Tag.ZERO, over.why)
    check("and it says the budget is the reason", "budget" in over.why)
    bad = parse_nesting(list("([)]"))
    check("mismatched brackets → MINUS (a real error, distinct from too-deep)",
          bad.tag is Tag.MINUS, bad.why)
    unclosed = parse_nesting(list("((("))
    check("unclosed brackets → MINUS", unclosed.tag is Tag.MINUS)
    refuses("a depth budget below 1 is refused",
            lambda: parse_nesting(clause, max_depth=0))

    print("\nZERO UNIFIES MISSING DATA AND MISSING BUDGET — the same honest middle")
    check("too-deep is ZERO, exactly as an unread tag is ZERO",
          over.tag is Tag.ZERO and Judgment.unknown().tag is Tag.ZERO)

    print("\nTHE GENERAL BREAKER: any non-halting computation is made total")
    # A deliberately non-terminating recursion, bounded to a ZERO halt.
    def forever(tick, depth=0):
        tick()
        return forever(tick, depth + 1)
    got = bounded(forever, max_steps=500)
    check("an infinite recursion halts at ZERO, not a crash", got.tag is Tag.ZERO, got.why)
    check("and it reports the budget it hit", "budget of 500" in got.why)

    # A computation that DOES finish returns PLUS with its result.
    def sum_to(tick, k):
        total = 0
        for i in range(k):
            tick()
            total += i
        return total
    done = bounded(sum_to, 100, max_steps=10_000)
    check("a terminating computation returns PLUS with its value",
          done.tag is Tag.PLUS and done.value == sum(range(100)), done.why)
    refuses("a zero-step budget is refused", lambda: bounded(sum_to, 5, max_steps=0))

    print("\nREADING DIRECTION IS A FINITE GROUP — Z2 × Z3, enumerated not searched")
    six = readings("ATGTTTGGA")
    check("there are exactly six readings", len(six) == 6, str(len(six)))
    check("the strands are reverse-complements",
          six["+0"] == "ATGTTTGGA" and six["−0"] == "TCCAAACAT", six["−0"])
    check("the three frames shift the start",
          six["+1"] == "TGTTTGGA" and six["+2"] == "GTTTGGA")
    check("the reading group is closed — there is no seventh",
          set(six) == {"+0", "+1", "+2", "−0", "−1", "−2"})

    print("\nAS A PIPELINE COMPONENT")
    from glossa.vocab import get
    from glossa.pipeline import Pipeline
    doc = Pipeline(get("text")).add(nesting(max_depth=10))("( ( a ) b )")
    check("the component tags the doc with the nesting verdict",
          doc.tags["nesting"].tag is Tag.PLUS, doc.tags["nesting"].why)
    check("and each token carries its depth",
          doc[0].tag("depth").value == 1 and doc[1].tag("depth").value == 2)

    print()
    if fails:
        print(f"FAILED — {len(fails)} of {n} check(s):")
        for f in fails:
            print("  · " + f)
        return 1
    print("GLOSSA BOUNDS WITNESS HOLDS — recursion is bounded to a three-valued\n"
          "  halt, MINUS (malformed) is kept distinct from ZERO (too deep), the\n"
          "  reading group is enumerated not searched, and meaning is refused\n"
          "  rather than hallucinated.")
    return 0


if __name__ == "__main__":
    sys.exit(_witness())
