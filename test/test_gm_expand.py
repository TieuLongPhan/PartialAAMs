import unittest
from synkit.IO.chem_converter import rsmi_to_graph

# Attempt importing gmapache and methods
try:
    import gmapache  # noqa: F401

    GM_AVAILABLE = True
    from partialaams.gm_expand import gm_extend_from_graph, gm_extend_aam_from_rsmi
except Exception:
    GM_AVAILABLE = False
    gm_extend_from_graph, gm_extend_aam_from_rsmi = None, None

from synkit.Chem.Reaction.aam_validator import AAMValidator


@unittest.skipUnless(GM_AVAILABLE, "gmapache not available — skipping GM test")
class TestGmExtendFromGraph(unittest.TestCase):
    def test_gm_extend_from_graph(self):
        """
        Tests the gm_extend_from_graph function to ensure it returns
        the correct extended RSMIs.
        """
        # Define the RSMI input
        rsmi = "CC[CH2:3][Cl:1].[NH3:2]>>CC[CH2:3][NH2:2].[Cl:1]"
        # Convert the RSMI to graph representations
        G, H = rsmi_to_graph(rsmi, drop_non_aam=False, use_index_as_atom_map=False)
        # Execute the graph expansion function
        result_smiles = gm_extend_from_graph(G, H, aam_key="atom_map")
        # Expected output
        expected = (
            "[Cl:1][CH2:5][CH2:4][CH3:3].[NH3:2]>>[Cl:1].[NH2:2][CH2:5][CH2:4][CH3:3]"
        )
        # Assert that the function returns the expected result
        self.assertTrue(AAMValidator.smiles_check(result_smiles, expected))

    def gm_extend_aam_from_rsmi(self):

        rsmi = "CC[CH2:3][Cl:1].[N:2]>>CC[CH2:3][N:2].[Cl:1]"

        # Execute the graph expansion function
        result_smiles = gm_extend_aam_from_rsmi(rsmi)
        # Expected output
        expected = [
            "[Cl:1][CH2:3][CH2:5][CH3:4].[NH3:2]>>[ClH:1].[NH2:2][CH2:3][CH2:5][CH3:4]"
        ]
        # Assert that the function returns the expected result
        self.assertTrue(AAMValidator.smiles_check(result_smiles, expected))


# Run the unittest
if __name__ == "__main__":
    unittest.main()
