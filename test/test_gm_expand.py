import unittest
from aamutils.utils import smiles_to_graph
from partialaams.gm_expand import gm_extend_from_graph


class TestGmExtendFromGraph(unittest.TestCase):
    def test_gm_extend_from_graph(self):
        """
        Tests the gm_extend_from_graph function to ensure it returns
        the correct extended RSMIs.
        """
        # Define the RSMI input
        rsmi = "CC[CH2:3][Cl:1].[N:2]>>CC[CH2:3][N:2].[Cl:1]"
        # Convert the RSMI to graph representations
        G, H = smiles_to_graph(rsmi)
        # Execute the graph expansion function
        result_smiles = gm_extend_from_graph(G, H)
        # Expected output
        expected = [
            "[Cl:1][CH2:3][CH2:5][CH3:4].[NH3:2]>>[ClH:1].[NH2:2][CH2:3][CH2:5][CH3:4]"
        ]
        # Assert that the function returns the expected result
        self.assertEqual(result_smiles, expected)


# Run the unittest
if __name__ == "__main__":
    unittest.main()
