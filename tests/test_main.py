import unittest
import pandas as pd

from unittest import TestCase
from pandas.testing import assert_series_equal, assert_frame_equal

from apps.main import *

if __name__ == "__main__":
    unittest.main()


class TestTrainTestSplitByOrder(TestCase):
    def test_train_test_split_by_order_array_0_len(self):
        array = []
        test_size = 0.2

        # wait the exception
        with self.assertRaises(ValueError) as context:
            train_test_split_by_order(array, test_size)

        expected = "The array must contain more than one element!"
        self.assertEqual(expected, str(context.exception))

    def test_train_test_split_by_order_array_1_len(self):
        array = [0]
        test_size = 0.2

        # wait the exception
        with self.assertRaises(ValueError) as context:
            train_test_split_by_order(array, test_size)

        expected = "The array must contain more than one element!"
        self.assertEqual(expected, str(context.exception))

    def test_train_test_split_by_order_test_size_0(self):
        array = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        test_size = 0.0

        # wait the exception
        with self.assertRaises(ValueError) as context:
            train_test_split_by_order(array, test_size)

        expected = "Invalid test size, it should be between 0.0 and 1.0!"
        self.assertEqual(expected, str(context.exception))

    def test_train_test_split_by_order_test_size_1(self):
        array = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        test_size = 1.0

        # wait the exception
        with self.assertRaises(ValueError) as context:
            train_test_split_by_order(array, test_size)

        expected = "Invalid test size, it should be between 0.0 and 1.0!"
        self.assertEqual(expected, str(context.exception))

    def test_train_test_split_by_order_array(self):
        array = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        test_size = 0.2

        actual_train, actual_test = train_test_split_by_order(array, test_size)
        expected_train, expected_test = [0, 1, 2, 3, 4, 5, 6, 7], [8, 9]
        self.assertEqual(expected_train, actual_train)
        self.assertEqual(expected_test, actual_test)

    def test_train_test_split_by_order_series(self):
        array = pd.Series([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        test_size = 0.2

        actual_train, actual_test = train_test_split_by_order(array, test_size)
        expected_train, expected_test = pd.Series([0, 1, 2, 3, 4, 5, 6, 7]), pd.Series([8, 9], index=[8, 9])

        assert_series_equal(expected_train, actual_train)
        assert_series_equal(expected_test, actual_test)

    def test_train_test_split_by_order_dataframe(self):
        array = pd.DataFrame([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        test_size = 0.2

        actual_train, actual_test = train_test_split_by_order(array, test_size)
        expected_train, expected_test = pd.DataFrame([0, 1, 2, 3, 4, 5, 6, 7]), pd.DataFrame([8, 9], index=[8, 9])

        assert_frame_equal(expected_train, actual_train)
        assert_frame_equal(expected_test, actual_test)
