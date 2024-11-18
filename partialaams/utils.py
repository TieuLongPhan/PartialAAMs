import numpy as np
import networkx as nx
import collections
from typing import List, Tuple, Optional
from copy import deepcopy
from rdkit.Chem.rdmolfiles import MolToSmiles
from aamutils.utils import get_beta_map, graph_to_mol


def get_aam_pairwise_indices(G: nx.Graph, H: nx.Graph) -> list:
    """
    Generates pairwise indices from two graphs based on atom-atom mapping (aam).

    Parameters:
    - G (nx.Graph): The first graph with 'aam' data on nodes.
    - H (nx.Graph): The second graph with 'aam' data on nodes.

    Returns:
    - list: A list of tuples, where each tuple contains corresponding
    node indices with the same 'aam'.
    """
    aam_to_index_G = {d["aam"]: n for n, d in G.nodes(data=True) if d["aam"] > 0}
    aam_to_index_H = {d["aam"]: n for n, d in H.nodes(data=True) if d["aam"] > 0}
    return [
        (aam_to_index_G[aam], aam_to_index_H[aam])
        for aam in aam_to_index_G
        if aam in aam_to_index_H
    ]


def set_aam(
    G_i: nx.Graph,
    H_i: nx.Graph,
    M: np.ndarray,
    beta_map: Optional[list] = None,
    aam_key: str = "aam",
) -> Tuple[nx.Graph, nx.Graph]:
    """
    Assigns atom-atom mappings (AAM) to two graphs based on an initial beta map and
    a transformation matrix, ensuring that the mappings are consistent across
    transformations represented by the matrix M.

    Parameters:
    - G_i (nx.Graph): Initial graph of reactants with pre-existing atom-atom mappings
    if available.
    - H_i (nx.Graph): Initial graph of products with pre-existing atom-atom mappings
    if available.
    - M (np.ndarray): A transformation matrix that defines how atom mappings should be
    transferred or transformed.
    - beta_map (Optional[list], default=None): A list of tuples defining direct mappings
    between atoms of G_i and H_i. If None, it will be calculated using `get_beta_map`.
    - aam_key (str, default='aam'): The key used to store atom-atom mapping information
    in node attributes.

    Returns:
    - Tuple[nx.Graph, nx.Graph]: A tuple of two graphs (G, H) where G is the
    modified reactant graph and H is the modified product graph with updated AAM
    based on the matrix M.
    """
    G, H = deepcopy(G_i), deepcopy(H_i)
    if beta_map is None:
        beta_map = get_beta_map(G, H)
    used_atom_numbers = [aam for _, _, aam in beta_map]

    aam_G = collections.defaultdict(lambda: -1)
    for bi, _, aam in beta_map:
        aam_G[bi] = aam

    # Assign AAMs to G based on beta_map or find new unique AAM numbers
    next_aam_nr = max(aam_G.values(), default=0) + 1
    for n in G:
        if aam_G[n] == -1:
            while next_aam_nr in used_atom_numbers:
                next_aam_nr += 1
            used_atom_numbers.append(next_aam_nr)
            aam_G[n] = next_aam_nr
        G.nodes[n][aam_key] = int(aam_G[n])

    # Transfer AAMs from G to H according to the transformation matrix M
    aam_G_array = np.array([aam_G[n] for n in sorted(G)])
    aam_H = np.dot(M.T, aam_G_array)
    for n, aam in zip(sorted(H), aam_H):
        H.nodes[n][aam_key] = int(aam)

    return G, H


def create_adjacency_matrix(pairwise_list: list) -> np.ndarray:
    """
    Creates an adjacency matrix for a graph based on pairwise node connections.

    Parameters:
    - pairwise_list (list): A list of tuples, each representing a pair of connected nodes.

    Returns:
    - np.ndarray: An adjacency matrix representing the graph.
    """
    size = max(x for pair in pairwise_list for x in pair) + 1
    adj_matrix = np.zeros((size, size), dtype=int)
    for v1, v2 in pairwise_list:
        adj_matrix[v1, v2] = 1
    return adj_matrix


def get_rsmi(G: nx.Graph, H: nx.Graph, M: np.ndarray) -> str:
    """
    Generates a reaction SMILES (RSMI) for the transformation represented
    by a single matrix.

    Parameters:
    - G (nx.Graph): Graph representing the reactants.
    - H (nx.Graph): Graph representing the products.
    - M (np.ndarray): A transformation matrix applying a mapping to the graphs.

    Returns:
    - str: A single reaction SMILES string representing the transformed reaction.
    """
    G_new, H_new = set_aam(G, H, M)
    r_mol = graph_to_mol(G_new)
    p_mol = graph_to_mol(H_new)
    result = "{}>>{}".format(MolToSmiles(r_mol), MolToSmiles(p_mol))
    return result


def get_list_of_rsmi(G: nx.Graph, H: nx.Graph, Ms: List[np.ndarray]) -> List[str]:
    """
    Generates a list of reaction SMILES (RSMI) strings for each transformation
    represented by the matrices in Ms.

    Parameters:
    - G (nx.Graph): Graph representing the reactants.
    - H (nx.Graph): Graph representing the products.
    - Ms (List[np.ndarray]): A list of transformation matrices.

    Returns:
    - List[str]: A list of reaction SMILES strings, each corresponding
    to a transformation.
    """
    return [get_rsmi(G, H, M) for M in Ms]
