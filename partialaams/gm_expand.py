import gmapache as gm
from aamutils.utils import smiles_to_graph
from partialaams.utils import (
    get_aam_pairwise_indices,
    create_adjacency_matrix,
    get_list_of_rsmi,
)


def gm_extend_from_graph(G, H):
    M = get_aam_pairwise_indices(G, H)
    results, _ = gm.maximum_connected_extensions(G, H, M, all_extensions=False)
    Ms = [create_adjacency_matrix(value) for value in results]
    return get_list_of_rsmi(G, H, Ms)


def gm_extend_aam_from_rsmi(rsmi: str) -> str:
    """
    Extends atom-atom mappings (AAM) from a reaction SMILES (RSMI) string,
    and returns the resulting reaction SMILES string.

    Parameters:
    - rsmi (str): A reaction SMILES string in the format 'reactant>>product'.

    Returns:
    - str: A reaction SMILES string with extended atom mappings.
    """
    G, H = smiles_to_graph(rsmi, sanitize=False)
    return gm_extend_from_graph(G, H)
