from __future__ import annotations

import random

import pytest


@pytest.fixture
def rng() -> random.Random:
    return random.Random(0)
