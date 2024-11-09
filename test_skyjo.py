#!/usr/local/bin/python3.13

from __future__ import annotations

import math
import unittest

from skyjo import *


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
    
    def test_add_assign(self) -> None:
        a = Average(9, 3)
        self.assertEqual(float(a), 3)
        b = a
        self.assertEqual(a, b)
        a += Average(1, 1)
        self.assertEqual(float(a), 2.5)
        self.assertNotEqual(a, b)


if __name__ == "__main__":
    unittest.main()
