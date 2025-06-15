import numpy as np
import collections
import networkx as nx
from copy import deepcopy
from typing import List, Tuple, Optional

from rdkit import Chem
from rdkit.Chem.rdmolfiles import MolToSmiles
from synkit.IO.graph_to_mol import GraphToMol
from partialaams.aam_utils import get_beta_map, graph_to_mol


def rename_node_attribute(graph: nx.Graph, old_attr: str, new_attr: str) -> None:
    """
    Rename node attribute 'old_attr' to 'new_attr' in the given graph.

    Args:
        graph (nx.Graph): The graph whose node attributes are to be renamed.
        old_attr (str): The existing attribute name to rename.
        new_attr (str): The new attribute name.

    The function modifies the graph in-place.
    """
    for _, data in graph.nodes(data=True):
        if old_attr in data:
            data[new_attr] = data.pop(old_attr)


def get_aam_pairwise_indices(G: nx.Graph, H: nx.Graph, aam_key: str) -> list:
    """
    Generates pairwise indices from two graphs based on atom-atom mapping (aam).

    Parameters:
    - G (nx.Graph): The first graph with 'aam' data on nodes.
    - H (nx.Graph): The second graph with 'aam' data on nodes.

    Returns:
    - list: A list of tuples, where each tuple contains corresponding
    node indices with the same 'aam'.
    """
    aam_to_index_G = {d[aam_key]: n for n, d in G.nodes(data=True) if d[aam_key] > 0}
    aam_to_index_H = {d[aam_key]: n for n, d in H.nodes(data=True) if d[aam_key] > 0}
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


def get_rsmi(G: nx.Graph, H: nx.Graph, M: np.ndarray, aam_key: str) -> str:
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
    G_new, H_new = _update_mapping(G, H, M, aam_key)
    r_mol = graph_to_mol(G_new)
    p_mol = graph_to_mol(H_new)
    result = "{}>>{}".format(MolToSmiles(r_mol), MolToSmiles(p_mol))
    return result


def get_list_of_rsmi(
    G: nx.Graph, H: nx.Graph, Ms: List[np.ndarray], aam_key
) -> List[str]:
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
    return [get_rsmi(G, H, M, aam_key) for M in Ms]


def its_decompose(
    its_graph: nx.Graph,
    nodes_share="typesGH",
    edges_share="order",
    use_atom_map: bool = False,
):
    """
    Decompose an ITS graph into two separate graphs G and H based on shared
    node and edge attributes.

    Parameters:
    - its_graph (nx.Graph): The integrated transition state (ITS) graph.
    - nodes_share (str): Node attribute key that stores tuples with node attributes
    or G and H.
    - edges_share (str): Edge attribute key that stores tuples with edge attributes
    for G and H.

    Returns:
    - Tuple[nx.Graph, nx.Graph]: A tuple containing the two graphs G and H.
    """
    G = nx.Graph()
    H = nx.Graph()

    # Decompose nodes
    for node, data in its_graph.nodes(data=True):
        if nodes_share in data:
            node_attr_g, node_attr_h = data[nodes_share]
            # Determine the value for 'atom_map'.
            atom_map_value = data.get("atom_map", node) if use_atom_map else node
            # Unpack node attributes for G
            G.add_node(
                node,
                element=node_attr_g[0],
                aromatic=node_attr_g[1],
                hcount=node_attr_g[2],
                charge=node_attr_g[3],
                neighbors=node_attr_g[4],
                atom_map=atom_map_value,
            )
            # Unpack node attributes for H
            H.add_node(
                node,
                element=node_attr_h[0],
                aromatic=node_attr_h[1],
                hcount=node_attr_h[2],
                charge=node_attr_h[3],
                neighbors=node_attr_h[4],
                atom_map=atom_map_value,
            )

    # Decompose edges
    for u, v, data in its_graph.edges(data=True):
        if edges_share in data:
            order_g, order_h = data[edges_share]
            if order_g > 0:  # Assuming 0 means no edge in G
                G.add_edge(u, v, order=order_g)
            if order_h > 0:  # Assuming 0 means no edge in H
                H.add_edge(u, v, order=order_h)

    return G, H


def _get_partial_aam(rc: nx.Graph, its: nx.Graph) -> str:
    """
    Generate a partial atom-atom mapping SMILES string from
    a reactant graph and an ITS graph.

    This function performs the following steps:
      1. Deep copies the ITS graph to avoid modifying the original.
      2. Iterates through each node in the copied ITS graph; if a node is not present
         in the reactant graph (rc), its 'atom_map' attribute is set to 0.
      3. Decomposes the modified graph into two subgraphs using `its_decompose`
         (one for retained mapping and one for partial mapping).
      4. Converts the resulting subgraphs into RDKit molecule objects with hydrogen count
         consideration using GraphToMol.
      5. Generates SMILES strings from both molecules and concatenates them in the format:
         "retained_smiles>>partial_smiles".

    Parameters:
      rc (nx.Graph): The reactant graph.
      its (nx.Graph): The integrated transition state (ITS) graph to decompose.

    Returns:
      str: A SMILES string in the format "retained_smiles>>partial_smiles".

    Raises:
      RuntimeError: If an error occurs during graph decomposition, molecule conversion,
                    or SMILES generation.
    """
    # Create a set of reactant nodes for efficient membership checking.
    rc_nodes = set(rc.nodes())

    # Deep copy the ITS graph to ensure the original graph remains unmodified.
    graph = deepcopy(its)

    # Update the 'atom_map' attribute for nodes not present in the reactant graph.
    for node, _ in graph.nodes(data=True):
        if node not in rc_nodes:
            graph.nodes[node]["atom_map"] = 0

    # Decompose the ITS graph into two subgraphs using the its_decompose function.
    try:
        retained_graph, partial_graph = its_decompose(graph, use_atom_map=True)
    except Exception as e:
        raise RuntimeError("Error during ITS graph decomposition: " + str(e)) from e

    # Convert the decomposed graphs into RDKit molecules.
    converter = GraphToMol()
    try:
        retained_mol = converter.graph_to_mol(retained_graph, use_h_count=True)
        partial_mol = converter.graph_to_mol(partial_graph, use_h_count=True)
    except Exception as e:
        raise RuntimeError(
            "Error converting graphs to RDKit molecules: " + str(e)
        ) from e

    # Generate SMILES strings for both molecules.
    try:
        retained_smiles = Chem.MolToSmiles(retained_mol)
        partial_smiles = Chem.MolToSmiles(partial_mol)
    except Exception as e:
        raise RuntimeError(
            "Error generating SMILES from RDKit molecules: " + str(e)
        ) from e

    # Return the concatenated SMILES string representing the mapping.
    return f"{retained_smiles}>>{partial_smiles}"


def _update_mapping(G, H, mapping, aam_key="atom_map"):
    """
    Update node attributes in graphs G and H based on a sequential mapping.

    This function first resets the node attribute specified by aam_key for
    every node in G and H to 0.
    Then, for each tuple (g_node, h_node) in the mapping list, it sets:
        G.nodes[g_node][aam_key] = i + 1
        H.nodes[h_node][aam_key] = i + 1
    where i is the index of the mapping (starting from 0).

    Parameters:
    -----------
    G : networkx.Graph
        Graph G whose nodes will be updated.
    H : networkx.Graph
        Graph H whose nodes will be updated.
    mapping : list of tuples
        A list of tuples (g_node, h_node) specifying the mapping between nodes in G and H.
    aam_key : str, optional (default='aam')
        The name of the node attribute to update.

    Returns:
    --------
    (G, H) : tuple
        The updated graphs.
    """
    # Reset the attribute for all nodes in G and H to 0.
    for node in G.nodes():
        G.nodes[node][aam_key] = 0
    for node in H.nodes():
        H.nodes[node][aam_key] = 0

    # Update the attribute to i+1 for nodes according to the mapping.
    for i, (g_node, h_node) in enumerate(mapping):
        value = i + 1
        if g_node in G:
            G.nodes[g_node][aam_key] = value
        else:
            print(f"Warning: Node {g_node} not found in graph G.")
        if h_node in H:
            H.nodes[h_node][aam_key] = value
        else:
            print(f"Warning: Node {h_node} not found in graph H.")

    return G, H
