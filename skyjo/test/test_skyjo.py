#!/usr/local/bin/python3.13

from __future__ import annotations

import gc
import sys
import unittest

import skyjo


class MockPlayer:
    """Used to test `Game` invocations."""


class TestGame(unittest.TestCase):
    """Verify binding behaviors."""

    def test_new_empty(self) -> None:
        with self.assertRaises(TypeError):
            skyjo.Game()

    def test_new_invalid(self) -> None:
        with self.assertRaises(TypeError):
            skyjo.Game(1)

    def test_new_few(self) -> None:
        with self.assertRaises(ValueError):
            skyjo.Game([MockPlayer()])

    def test_new_many(self) -> None:
        with self.assertRaises(ValueError):
            skyjo.Game([MockPlayer() for _ in range(9)])

    def test_new_refcount(self) -> None:
        players = [MockPlayer() for _ in range(3)]
        game = skyjo.Game(players)
        self.assertEqual(sys.getrefcount(players[0]), 3)
        self.assertEqual(sys.getrefcount(players[1]), 3)
        self.assertEqual(sys.getrefcount(players[2]), 3)
        del game
        gc.collect()
        self.assertEqual(sys.getrefcount(players[0]), 2)
        self.assertEqual(sys.getrefcount(players[1]), 2)
        self.assertEqual(sys.getrefcount(players[2]), 2)


if __name__ == "__main__":
    unittest.main()
