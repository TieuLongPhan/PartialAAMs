import networkx as nx
import rdkit.Chem.rdmolfiles as rdmolfiles
from aamutils.algorithm.ilp import expand_partial_aam_balanced
from partialaams.aam_utils import set_aam, graph_to_mol, smiles_to_graph


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


def extend_aam_from_rsmi(rsmi: str) -> str:
    """
    Extends atom-atom mappings (AAM) from a reaction SMILES (RSMI) string,
    and returns the resulting reaction SMILES string.

    Parameters:
    - rsmi (str): A reaction SMILES string in the format 'reactant>>product'.

    Returns:
    - str: A reaction SMILES string with extended atom mappings.
    """
    G, H = smiles_to_graph(rsmi, sanitize=False)
    return extend_aam_from_graph(G, H)
