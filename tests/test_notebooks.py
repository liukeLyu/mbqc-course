"""Execute every lesson notebook (slow: run with `pytest -m slow`)."""

from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")
nbclient = pytest.importorskip("nbclient")

NOTEBOOKS = sorted((Path(__file__).parent.parent / "notebooks").glob("*.ipynb"))


@pytest.mark.slow
@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.stem)
def test_notebook_runs(path):
    nb = nbformat.read(path, as_version=4)
    nbclient.NotebookClient(nb, timeout=600, resources={"metadata": {"path": str(path.parent)}}).execute()
