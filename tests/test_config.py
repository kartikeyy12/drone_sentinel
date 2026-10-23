"""The YAML config must not repeat a key: PyYAML keeps the last duplicate and drops the first without a word."""
from __future__ import annotations

import yaml

from sentinel.config import DEFAULT_CONFIG, load_config


class _Strict(yaml.SafeLoader):
    pass


def _no_duplicates(loader, node, deep=False):
    seen = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in seen:
            raise ValueError(f"duplicate key {key!r} (line {key_node.start_mark.line + 1})")
        seen.add(key)
    return yaml.SafeLoader.construct_mapping(loader, node, deep)


_Strict.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _no_duplicates)


def test_default_config_has_no_duplicate_keys():
    with open(DEFAULT_CONFIG, encoding="utf-8") as f:
        yaml.load(f, _Strict)


def test_strict_loader_does_catch_a_duplicate():
    import pytest
    with pytest.raises(ValueError, match="duplicate key 'a'"):
        yaml.load("a: 1\nb: 2\na: 3\n", _Strict)


def test_the_blocks_the_milestones_read_exist():
    cfg = load_config()
    for k in ("m2", "m3", "m4"):
        assert k in cfg
    assert {"default", "grid", "silence_gate_rms"} <= set(cfg["m3"])
    assert {"robustness", "final", "operating", "latency", "positive_classes"} <= set(cfg["m4"])
