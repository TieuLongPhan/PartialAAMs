import unittest

from partialaams.aam_expand import partial_aam_extension_from_smiles
from synkit.Graph.ITS.aam_validator import AAMValidator


class TestPartialAAMExtension(unittest.TestCase):

    def setUp(self):
        """
        This method is called before each test. You can use it to set up
        any state you want to share across tests.
        """
        self.rsmi = "[CH3][CH:1]=[CH2:2].[H:3][H:4]>>[CH3][CH:1]([H:3])[CH2:2][H:4]"
        self.expected = (
            "[CH2:1]=[CH:2][CH3:3].[H:4][H:5]>>" + "[CH2:1]([CH:2]([CH3:3])[H:5])[H:4]"
        )

    def test_ilp_extension(self):
        """
        Test extension using the ILP-based method.
        """
        result = partial_aam_extension_from_smiles(self.rsmi, method="ilp")
        self.assertTrue(AAMValidator.smiles_check(result, self.expected))

    def test_gm_extension(self):
        """
        Test extension using the GM-based method.
        """
        result = partial_aam_extension_from_smiles(self.rsmi, method="gm")
        self.assertTrue(AAMValidator.smiles_check(result, self.expected))

    def test_syn_extension(self):
        """
        Test extension using the SynAAM-based method.
        """
        result = partial_aam_extension_from_smiles(self.rsmi, method="syn")
        self.assertTrue(AAMValidator.smiles_check(result, self.expected))

    def test_invalid_method(self):
        """
        Test that an invalid method raises a ValueError.
        """
        with self.assertRaises(ValueError):
            partial_aam_extension_from_smiles(self.rsmi, method="invalid_method")


if __name__ == "__main__":
    unittest.main()
