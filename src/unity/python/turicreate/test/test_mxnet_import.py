import unittest
import sys


class MXNetImportTest(unittest.TestCase):

    def test_mxnet_import(self):

        self.assertTrue("mxnet" not in sys.modules)

        import turicreate

        self.assertTrue("mxnet" not in sys.modules)
