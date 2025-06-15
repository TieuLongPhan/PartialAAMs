import unittest
from partialaams.aam_utils import smiles_to_graph
from partialaams.ilp_expand import extend_aam_from_rsmi, extend_aam_from_graph
from synkit.Chem.Reaction.aam_validator import AAMValidator


class TesILPExpand(unittest.TestCase):
    def test_ilp_extend_from_graph(self):
        """
        Tests the gm_extend_from_graph function to ensure it returns
        the correct extended RSMIs.
        """
        # Define the RSMI input
        rsmi = "CC[CH2:3][Cl:1].[NH3:2]>>CC[CH2:3][NH2:2].[Cl:1]"
        # Convert the RSMI to graph representations
        G, H = smiles_to_graph(rsmi)
        # Execute the graph expansion function
        result_smiles = extend_aam_from_graph(G, H, aam_key="aam")
        # Expected output
        expected = (
            "[Cl:1][CH2:3][CH2:5][CH3:4].[NH3:2]>>[ClH:1].[NH2:2][CH2:3][CH2:5][CH3:4]"
        )
        # Assert that the function returns the expected result
        self.assertTrue(AAMValidator.smiles_check(result_smiles, expected))

    def ilp_extend_aam_from_rsmi(self):

        rsmi = "CC[CH2:3][Cl:1].[N:2]>>CC[CH2:3][N:2].[Cl:1]"

        # Execute the graph expansion function
        result_smiles = extend_aam_from_rsmi(rsmi)
        # Expected output
        expected = [
            "[Cl:1][CH2:3][CH2:5][CH3:4].[NH3:2]>>[ClH:1].[NH2:2][CH2:3][CH2:5][CH3:4]"
        ]
        # Assert that the function returns the expected result
        self.assertTrue(AAMValidator.smiles_check(result_smiles, expected))


# Run the unittest
if __name__ == "__main__":
    unittest.main()
