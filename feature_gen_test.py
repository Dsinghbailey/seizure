import numpy as np
import unittest
import pandas as pd

import feature_gen


class SmoothTest(unittest.TestCase):
    def test_smooth(self):
        a = np.array([1,2,3,4,9,9,9,9] + 8 * [42])
        self.assertAllEqual(
            np.array([1,2,3.5,9,42]),
            feature_gen.smooth_fft(a, stride=1))

    def assertAllEqual(self, xs, ys):
        for x, y in zip(xs, ys):
            self.assertEqual(x, y)

    def test_cant_make_array_from_sequence(self):
        self.assertEqual(
            (),
            np.array(i for i in [1,2,3]).shape)


class FFTTest(unittest.TestCase):

    def test_fft(self):
        data = pd.DataFrame(np.transpose(np.array([
            [1,2,3,4,5,6],
            [9,9,9,9,9,9]
        ])))

        for features in feature_gen.fft(data):
            self.assertEqual(1, len(features.shape))
            self.assertTrue(0 < features.shape[0])



if __name__ == '__main__':
    unittest.main()
