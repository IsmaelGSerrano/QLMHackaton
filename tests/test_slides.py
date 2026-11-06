"""The slide deck (docs/slides/index.html) cites figures and result files from the repo.

These tests keep it honest: every local file it links must exist, and the main evidence figures must be on a slide.
"""

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DECK = REPO / "docs" / "slides" / "index.html"


def _local_refs(html: str) -> list[str]:
    refs = re.findall(r'(?:src|href)="([^"#]+)"', html)
    return [r for r in refs if not re.match(r"^(https?:|mailto:|data:)", r)]


def test_every_local_link_in_the_deck_exists():
    html = DECK.read_text(encoding="utf-8")
    missing = [r for r in _local_refs(html) if not (DECK.parent / r).resolve().exists()]
    assert not missing, f"deck links to missing files: {missing}"


def test_deck_shows_the_evidence_figures():
    refs = {(DECK.parent / r).resolve() for r in _local_refs(DECK.read_text(encoding="utf-8"))}
    for evidence in [
        "results/figures/direction_ablation.png",
        "results/figures/qgnn_learning_curve.png",
        "reports/HI-Small/figures/motif_examples.png",
        "results/hardware/wp6_garnet_summary.md",
    ]:
        assert (REPO / evidence).resolve() in refs, f"{evidence} is not on a slide"
