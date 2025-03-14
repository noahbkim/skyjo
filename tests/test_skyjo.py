#!/usr/local/bin/python3.13

from __future__ import annotations

import math
import unittest

from skyjo.skyjo import *


class TestAverage(unittest.TestCase):
    """Ensure arithmetic works."""

    def test_construct(self) -> None:
        average = Average()
        self.assertTrue(math.isnan(float(average)))

    def test_construct_values(self) -> None:
        average = Average(100, 10)
        self.assertEqual(float(average), 10)

    def test_add_float(self) -> None:
        average = Average()
        average += 1
        self.assertEqual(float(average), 1)

    def test_add_float_right(self) -> None:
        average = Average()
        average = 1 + average
        self.assertEqual(float(average), 1)

    def test_add_float_many(self) -> None:
        average = Average()
        average += 1
        average += 2
        average += 3
        self.assertEqual(float(average), 2)

    def test_add_average(self) -> None:
        a = Average(9, 3)
        b = Average(1, 1)
        self.assertEqual(float(a + b), 2.5)

    def test_add_immutable(self) -> None:
        a = Average(9, 3)
        self.assertEqual(float(a), 3)
        b = a
        self.assertEqual(a, b)
        a += Average(1, 1)
        self.assertEqual(float(a), 2.5)
        self.assertNotEqual(a, b)


class TestCards(unittest.TestCase):
    """Ensure draws and discards are successful."""

    def test_construct(self) -> None:
        Cards()

    def test_last_discard(self) -> None:
        cards = Cards()
        self.assertEqual(cards.last_discard, DECK[0])
        cards._discard_card(0)
        self.assertEqual(cards.last_discard, 0)

    def test_average_deck_card_value(self) -> None:
        cards = Cards()
        self.assertEqual(cards.average_deck_card_value, sum(DECK) / len(DECK))

    def test__next_draw_card(self) -> None:
        cards = Cards()
        self.assertEqual(cards._next_draw_card, DECK[1])

    def test__draw_card(self) -> None:
        cards = Cards()
        self.assertEqual(cards._next_draw_card, DECK[1])
        cards._draw_card()
        self.assertEqual(cards._next_draw_card, DECK[2])

    def test__draw_card_restock(self) -> None:
        cards = Cards()
        for _ in range(1, len(DECK) * 2 + 1):
            cards._discard_card(cards._draw_card())
            cards.validate()

    def test__replace_discard_card(self) -> None:
        cards = Cards()
        self.assertEqual(cards._replace_discard_card(0), DECK[0])
        self.assertEqual(cards.last_discard, 0)


class TestFinger(unittest.TestCase):
    def test__flipped_visible(self) -> None:
        finger = Finger(_card=0)
        finger._flip_card()
        self.assertEqual(finger.is_visible, True)

    def test__init_not_visible(self) -> None:
        finger = Finger(_card=0)
        self.assertEqual(finger.is_flipped, False)
        self.assertEqual(finger.is_visible, False)

    def test__flipped_cleared_card_not_visible(self) -> None:
        finger = Finger(_card=0)
        finger._flip_card()
        finger._clear()
        self.assertEqual(finger.is_visible, False)


if __name__ == "__main__":
    unittest.main()
