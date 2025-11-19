import unittest
from partialaams.extender import Extender
from synkit.Chem.Reaction.aam_validator import AAMValidator

try:
    import gmapache  # noqa: F401

    GM_AVAILABLE = True
except Exception:
    GM_AVAILABLE = False


class TestExtender(unittest.TestCase):

    def test_extender_nx_rsmi(self):

        rsmi = "CC[CH2:3][Cl:1].[N:2]>>CC[CH2:3][N:2].[Cl:1]"

        # Execute the graph expansion function
        result_smiles = Extender().fit(rsmi, use_gm=False)
        # Expected output
        expected = (
            "[CH3:1][CH2:2][CH2:3][Cl:5].[N:4]>>[CH3:1][CH2:2][CH2:3][N:4].[Cl:5]"
        )
        # Assert that the function returns the expected result
        self.assertTrue(AAMValidator.smiles_check(result_smiles, expected))

    @unittest.skipUnless(GM_AVAILABLE, "gmapache not available — skipping GM test")
    def test_extender_gm_rsmi(self):

        rsmi = "CC[CH2:3][Cl:1].[N:2]>>CC[CH2:3][N:2].[Cl:1]"

        # Execute the graph expansion function
        result_smiles = Extender().fit(rsmi, use_gm=True)
        # Expected output
        expected = (
            "[CH3:1][CH2:2][CH2:3][Cl:5].[N:4]>>[CH3:1][CH2:2][CH2:3][N:4].[Cl:5]"
        )
        # Assert that the function returns the expected result
        self.assertTrue(AAMValidator.smiles_check(result_smiles, expected))


# Run the unittest
if __name__ == "__main__":
    unittest.main()
