"""kev.benchmark under the serving context.

    python -m signrule.train.kev_bench <kev.benchmark args>


kev.benchmark encodes `--data` records under the 384-token training context and skips longer
states (rejected.json), while the server accepts states up to kev.model.SERVE_MAX_STATE. The
longest Austrian extracts (~10 % of every AT part) were therefore never scored. Here every record is
encoded as the server would (kev.suite.SERVING_CONTEXT, long rows on the memory-efficient kernels):
we measure what we serve. Found 2026-10-07 (plan-16).
"""

from __future__ import annotations

import sys
from typing import Any

import kev.benchmark as benchmark
from kev.suite import SERVING_CONTEXT


def main() -> Any:
    benchmark.CONTEXT = SERVING_CONTEXT
    return benchmark.main()


if __name__ == "__main__":
    sys.exit(main())
