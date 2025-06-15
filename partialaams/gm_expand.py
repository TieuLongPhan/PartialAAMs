import gmapache as gm
from synkit.IO.chem_converter import smiles_to_graph

from partialaams.utils import (
    get_aam_pairwise_indices,
    # get_list_of_rsmi,
    _update_mapping,
)
from rdkit import Chem
from synkit.IO.graph_to_mol import GraphToMol


def gm_extend_from_graph(G, H, balance=True, aam_key="atom_map"):
    if balance:
        M = get_aam_pairwise_indices(G, H, aam_key)
        Ms, _ = gm.search_stable_extension(
            G, H, M, node_labels=True, edge_labels=True, all_extensions=False
        )
    else:
        if G.number_of_nodes() > H.number_of_nodes():
            M = get_aam_pairwise_indices(G, H)
            Ms, _ = gm.search_maximum_common_anchored_subgraphs(
                G,
                H,
                M,
                node_labels=True,
                edge_labels=True,
                all_extensions=True,
                reachability=True,
            )

        elif G.number_of_nodes() < H.number_of_nodes():
            M = get_aam_pairwise_indices(H, G)
            Ms, _ = gm.search_maximum_common_anchored_subgraphs(
                H,
                G,
                M,
                node_labels=True,
                edge_labels=True,
                all_extensions=True,
                reachability=True,
            )
            Ms = [[(h, g) for g, h in mapping] for mapping in Ms]

    # Update the mapping of the original graphs.
    G_new, H_new = _update_mapping(G, H, Ms[0], aam_key="atom_map")

    # Convert updated graphs back to molecular objects.
    reactant_mol = GraphToMol().graph_to_mol(G_new, use_h_count=True)
    product_mol = GraphToMol().graph_to_mol(H_new, use_h_count=True)

    # Generate and return the new reaction SMILES string.
    return f"{Chem.MolToSmiles(reactant_mol)}>>{Chem.MolToSmiles(product_mol)}"


def gm_extend_aam_from_rsmi(rsmi: str, balance=True, aam_key="atom_map") -> str:
    """
    Extends atom-atom mappings (AAM) from a reaction SMILES (RSMI) string,
    and returns the resulting reaction SMILES string.

    Parameters:
    - rsmi (str): A reaction SMILES string in the format 'reactant>>product'.

    Returns:
    - str: A reaction SMILES string with extended atom mappings.
    """
    # Convert SMILES to graphs using the provided chem converter.
    r, p = rsmi.split(">>")
    G = smiles_to_graph(r, drop_non_aam=False, use_index_as_atom_map=False)
    H = smiles_to_graph(p, drop_non_aam=False, use_index_as_atom_map=False)
    return gm_extend_from_graph(G, H, balance, aam_key)
