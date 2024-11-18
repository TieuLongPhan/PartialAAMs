import rdkit.Chem.rdmolfiles as rdmolfiles
from aamutils.utils import set_aam, graph_to_mol
from aamutils.algorithm.ilp import expand_partial_aam_balanced
import networkx as nx


def extend_aam_from_graph(G: nx.Graph, H: nx.Graph) -> str:
    """
    Extends atom-atom mappings (AAM) from two input graphs,
    G (reactants) and H (products),by solving the AAM problem using an
    Integer Linear Programming (ILP) approach and generating
    a reaction SMILES (RSMI) string.

    Parameters:
    - G (nx.Graph): Graph representing the reactants.
    Nodes should contain necessary attributes for AAM.
    - H (nx.Graph): Graph representing the products.
    Nodes should contain necessary attributes for AAM.

    Returns:
    - str: A reaction SMILES string (RSMI) in the format 'reactant>>product'
    with extended atom mappings.
    """
    # Solve the partial AAM problem using ILP and retrieve the mapping matrix
    M, _, _ = expand_partial_aam_balanced(G, H)

    # Apply the AAM matrix to the graphs
    set_aam(G, H, M)

    # Convert the modified graphs back to RDKit molecules
    r_mol = graph_to_mol(G)
    p_mol = graph_to_mol(H)

    # Generate reaction SMILES string from the RDKit molecules
    result_smiles = "{}>>{}".format(
        rdmolfiles.MolToSmiles(r_mol), rdmolfiles.MolToSmiles(p_mol)
    )
    return result_smiles
