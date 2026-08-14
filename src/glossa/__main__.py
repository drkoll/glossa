import sys

from glossa.bounds import _witness as bounds_w
from glossa.core import _witness as core_w
from glossa.pipeline import _witness as pipe_w
from glossa.vocab import _witness as vocab_w


def _entry() -> int:
    for w in (core_w, vocab_w, pipe_w, bounds_w):
        rc = w()
        if rc != 0:
            return rc
    return 0


if __name__ == "__main__":
    sys.exit(_entry())
