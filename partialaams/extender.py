import networkx as nx
import gmapache as gm


from synkit.IO.chem_converter import smiles_to_graph
from operator import eq
from typing import Callable, Optional, Dict, Any, List, Tuple
from networkx.algorithms.isomorphism import GraphMatcher
from networkx.algorithms.isomorphism import generic_node_match, generic_edge_match
from partialaams.utils import _update_mapping
from synkit.IO.graph_to_mol import GraphToMol
from rdkit import Chem


def remove_node_attributes_by_list(graph: nx.Graph, attributes: List[str]) -> nx.Graph:
    """
    Remove specified attributes from all nodes in a NetworkX graph.

    Parameters:
        graph (nx.Graph): The NetworkX graph whose nodes' attributes will be modified.
        attributes (List[str]): A list of attribute names to be removed.

    Returns:
        nx.Graph: The graph with the specified attributes removed from each node.
    """
    for node in graph.nodes():
        for attr in attributes:
            if attr in graph.nodes[node]:
                del graph.nodes[node][attr]
    return graph


class Extender:
    """
    A class for extending chemical reaction graphs by comparing and mapping reaction components.

    This class provides methods to identify nodes with non-zero 'atom_map' attributes,
    remove specific edges from graphs, determine graph isomorphism, convert mappings to tuple
    lists, and update mappings between graphs to generate an extended reaction SMILES.
    """

    def __init__(self) -> None:
        pass

    @staticmethod
    def _get_node_with_attribute(graph: nx.Graph) -> List[Any]:
        """
        Extract nodes from the graph that have a non-zero 'atom_map' attribute.

        Parameters:
            graph (nx.Graph): A NetworkX graph where nodes include an 'atom_map' attribute.

        Returns:
            List[Any]: A list of node identifiers with a non-zero 'atom_map' attribute.
        """
        nodes_with_attribute = []
        for node, attrs in graph.nodes(data=True):
            if attrs.get("atom_map", 0) != 0:
                nodes_with_attribute.append(node)
        return nodes_with_attribute

    @staticmethod
    def _remove_edges_between_nodes(G: nx.Graph, node_list: List[Any]) -> nx.Graph:
        """
        Remove all edges from graph G that connect any two nodes in node_list.

        Parameters:
            G (nx.Graph): The input graph.
            node_list (List[Any]): A list of node identifiers.

        Returns:
            nx.Graph: The graph with the specified edges removed.
        """
        node_set = set(node_list)
        # Collect edges where both endpoints are in the node_set.
        edges_to_remove = [
            (u, v) for u, v in G.edges() if u in node_set and v in node_set
        ]
        G.remove_edges_from(edges_to_remove)
        return G

    @staticmethod
    def dict_to_tuple_list(
        d: Dict[Any, Any], sort_by_key: bool = False, sort_by_value: bool = False
    ) -> List[Tuple[Any, Any]]:
        """
        Convert a dictionary to a list of tuples.

        Parameters:
            d (Dict[Any, Any]): The input dictionary.
            sort_by_key (bool): If True, sort the output list by dictionary keys.
            sort_by_value (bool): If True, sort the output list by dictionary values.

        Returns:
            List[Tuple[Any, Any]]: A list of tuples representing the dictionary items.
        """
        tuple_list = list(d.items())
        if sort_by_key:
            tuple_list.sort(key=lambda item: item[0])
        elif sort_by_value:
            tuple_list.sort(key=lambda item: item[1])
        return tuple_list

    @staticmethod
    def graph_isomorphism(
        graph_1: nx.Graph,
        graph_2: nx.Graph,
        node_match: Optional[Callable[[Dict[str, Any], Dict[str, Any]], bool]] = None,
        edge_match: Optional[Callable[[Dict[str, Any], Dict[str, Any]], bool]] = None,
        use_defaults: bool = True,
    ) -> Optional[Dict[Any, Any]]:
        """
        Determines if two graphs are isomorphic and returns the mapping from graph_1 to graph_2.

        If the graphs are isomorphic, returns a dictionary mapping nodes in graph_1 to
        corresponding nodes in graph_2. Otherwise, returns None.

        Parameters:
            graph_1 (nx.Graph): The first graph to compare.
            graph_2 (nx.Graph): The second graph to compare.
            node_match (Optional[Callable]): A function for comparing node attribute dictionaries.
                                             If None and use_defaults is True, a default match based on
                                             "element" and "atom_map" attributes is used.
            edge_match (Optional[Callable]): A function for comparing edge attribute dictionaries.
                                             If None and use_defaults is True, a default match based on the
                                             "order" attribute is used.
            use_defaults (bool): Whether to use default matching settings (defaults to True).

        Returns:
            Optional[Dict[Any, Any]]: Mapping of nodes from graph_1 to graph_2 if isomorphic; otherwise, None.
        """
        if use_defaults:
            node_label_names = ["element", "atom_map", 'hcount']
            node_label_default = ["*", 0, 0]
            edge_attribute = "order"

            if node_match is None:
                node_match = generic_node_match(
                    node_label_names, node_label_default, [eq] * len(node_label_names)
                )
            if edge_match is None:
                edge_match = generic_edge_match(edge_attribute, 1, eq)

        matcher = GraphMatcher(
            graph_1, graph_2, node_match=node_match, edge_match=edge_match
        )
        if matcher.is_isomorphic():
            return matcher.mapping
        else:
            return None

    def fit(self, rsmi: str, use_gm=False) -> str:
        """
        Extends a reaction SMILES string by updating the mapping of atoms between reactant and product graphs.

        The process involves:
          1. Converting reactant and product SMILES strings into graphs.
          2. Extracting nodes with a non-zero 'atom_map' attribute.
          3. Removing edges between these selected nodes.
          4. Determining graph isomorphism to obtain a mapping.
          5. Updating the original graphs using the new mapping.
          6. Converting the updated graphs back into molecular objects and generating the new reaction SMILES.

        Parameters:
            rsmi (str): A reaction SMILES string in the format "reactant>>product".

        Returns:
            str: The extended reaction SMILES string after mapping update.

        Raises:
            ValueError: If the input reaction SMILES format is incorrect or if the graphs are not isomorphic.
        """
        try:
            reactant_smiles, product_smiles = rsmi.split(">>")
        except ValueError as e:
            raise ValueError(
                "Input reaction SMILES must be in the format 'reactant>>product'"
            ) from e

        # Convert SMILES to graphs using the provided chem converter.
        reactant_graph = smiles_to_graph(
            reactant_smiles, drop_non_aam=False, use_index_as_atom_map=False
        )
        product_graph = smiles_to_graph(
            product_smiles, drop_non_aam=False, use_index_as_atom_map=False
        )

        # Extract nodes with non-zero 'atom_map' attribute.
        reactant_nodes = self._get_node_with_attribute(reactant_graph)
        product_nodes = self._get_node_with_attribute(product_graph)

        # Remove edges between these nodes to generate modified (prime) graphs.
        reactant_prime = self._remove_edges_between_nodes(
            reactant_graph.copy(), reactant_nodes
        )
        product_prime = self._remove_edges_between_nodes(
            product_graph.copy(), product_nodes
        )
        if use_gm:
            reactant_prime = remove_node_attributes_by_list(
                reactant_prime, ["aromatic", "charge", "neighbors"]
            )
            product_prime = remove_node_attributes_by_list(
                product_prime, ["aromatic", "charge", "neighbors"]
            )
            mapping_tuples, _ = gm.search_isomorphisms(
                reactant_prime,
                product_prime,
                all_isomorphisms=False,
                node_labels=True,
                edge_labels=True,
            )
            mapping_tuples = mapping_tuples[0]
        else:
            # Compute isomorphism mapping between the modified graphs.
            mapping_dict = self.graph_isomorphism(reactant_prime, product_prime)
            if mapping_dict is None:
                raise ValueError("Graphs are not isomorphic; cannot update mapping.")

            # Convert mapping dictionary to a list of tuples.
            mapping_tuples = self.dict_to_tuple_list(mapping_dict)

        # Update the mapping of the original graphs.
        G_new, H_new = _update_mapping(
            reactant_graph, product_graph, mapping_tuples, aam_key="atom_map"
        )

        # Convert updated graphs back to molecular objects.
        reactant_mol = GraphToMol().graph_to_mol(G_new, use_h_count=True)
        product_mol = GraphToMol().graph_to_mol(H_new, use_h_count=True)

        # Generate and return the new reaction SMILES string.
        return f"{Chem.MolToSmiles(reactant_mol)}>>{Chem.MolToSmiles(product_mol)}"
