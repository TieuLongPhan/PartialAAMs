import unittest
from aamutils.utils import smiles_to_graph
from partialaams.aam_expand import extend_aam_from_graph


class TestExtendAAMFromGraph(unittest.TestCase):
    def test_extend_aam_from_graph(self):
        """
        Tests the extend_aam_from_graph function to ensure it correctly extends
        atom-atom mappings and returns the expected reaction SMILES.
        """
        # Define the RSMI input
        rsmi = "CC[CH2:3][Cl:1].[N:2]>>CC[CH2:3][N:2].[Cl:1]"
        # Convert the RSMI to graph representations
        G, H = smiles_to_graph(rsmi)
        # Execute the AAM extension function
        result_smiles = extend_aam_from_graph(G, H)
        # Expected output
        expected = (
            "[Cl:1][CH2:3][CH2:5][CH3:4].[NH3:2]>>[ClH:1].[NH2:2][CH2:3][CH2:5][CH3:4]"
        )
        # Assert that the function returns the expected result
        self.assertEqual(result_smiles, expected)


# Run the unittest
if __name__ == "__main__":
    unittest.main()
