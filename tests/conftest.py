import pytest

from helpers_stream import build_toy


@pytest.fixture(scope="session")
def toy():
    """(cfg, model): a small model on real features of synthetic audio, built once per test session."""
    return build_toy()


@pytest.fixture(scope="session")
def toy_full():
    """(cfg, model, dataset, folds) of the toy model."""
    from helpers_stream import build_toy_full
    return build_toy_full()
