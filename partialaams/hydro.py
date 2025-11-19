import networkx as nx
from typing import List, Tuple, Dict

from synkit.Graph.Hyrogen._misc import check_hcount_change
from synkit.Graph.ITS.its_decompose import get_rc, its_decompose

import itertools
import networkx as nx
from copy import deepcopy, copy
from joblib import Parallel, delayed
from typing import Dict, List, Tuple, Iterable, Optional

from synkit.IO.debug import setup_logging
from synkit.Graph.Feature.wl_hash import WLHash
from synkit.Graph.ITS.its_construction import ITSConstruction
from synkit.Graph.ITS.its_decompose import get_rc, its_decompose
from synkit.Graph.Hyrogen._misc import (
    check_hcount_change,
    check_explicit_hydrogen,
    get_priority,
    check_equivariant_graph,
)
import logging


import importlib.util
import networkx as nx
from operator import eq
from collections import OrderedDict
from typing import List, Set, Dict, Any, Tuple, Optional, Callable
from networkx.algorithms.isomorphism import generic_node_match, generic_edge_match

from synkit.Rule.Modify.rule_utils import strip_context
from synkit.Graph.Matcher.graph_morphism import graph_isomorphism
from synkit.Graph.Matcher.graph_matcher import GraphMatcherEngine
import gmapache as gmap

if importlib.util.find_spec("mod") is not None:
    gm = GraphMatcherEngine(backend="mod")


class GraphCluster:
    def __init__(
        self,
        node_label_names: List[str] = ["element", "charge"],
        node_label_default: List[Any] = ["*", 0],
        edge_attribute: str = "order",
        backend: str = "nx",
    ):
        """Initializes the GraphCluster with customization options for node and
        edge matching functions. This class is designed to facilitate
        clustering of graph nodes and edges based on specified attributes and
        their matching criteria.

        Parameters:
        - node_label_names (List[str]): A list of node attribute names to be considered
          for matching. Each attribute name corresponds to a property of the nodes in the
          graph. Default values provided.
        - node_label_default (List[Any]): Default values for each of the node attributes
          specified in `node_label_names`. These are used where node attributes are missing.
          The length and order of this list should match `node_label_names`.
        - edge_attribute (str): The name of the edge attribute to consider for matching
          edges. This attribute is used to assess edge similarity.

        Raises:
        - ValueError: If the lengths of `node_label_names` and `node_label_default` do not
          match.
        """
        self.backend = backend.lower()
        available = self.available_backends()
        if self.backend not in available:
            if self.backend == "mod":
                raise ImportError("MOD is not installed")
            raise ValueError(f"Unsupported backend: {backend!r}")

        if len(node_label_names) != len(node_label_default):
            raise ValueError(
                "The lengths of `node_label_names` and `node_label_default` must match."
            )
        if backend == "nx":
            self.nodeLabelNames = node_label_names
            self.nodeLabelDefault = node_label_default
            self.edgeAttribute = edge_attribute
            self.nodeMatch = generic_node_match(
                self.nodeLabelNames,
                self.nodeLabelDefault,
                [eq for _ in node_label_names],
            )
            self.edgeMatch = generic_edge_match(self.edgeAttribute, 1, eq)
        else:
            self.nodeMatch = None
            self.edgeMatch = None

    def available_backends(self) -> List[str]:
        """
        Return available backends: always includes 'nx'; adds 'mode' if the 'mod' package is installed.
        """
        import importlib.util

        backends = ["nx"]
        # Check if 'mod' package is importable without executing it
        if importlib.util.find_spec("mod") is not None:
            backends.append("mod")
        return backends

    def iterative_cluster(
        self,
        rules: List[str],
        attributes: Optional[List[Any]] = None,
        nodeMatch: Optional[Callable] = None,
        edgeMatch: Optional[Callable] = None,
        aut: Optional[Any] = None,
    ) -> Tuple[List[Set[int]], Dict[int, int]]:
        """Clusters rules based on their similarities, which could include
        structural or attribute-based similarities depending on the given
        attributes.

        Parameters:
        - rules (List[str]): List of rules, potentially serialized strings of rule
          representations.
        - attributes (Optional[List[Any]]): Attributes associated with each rule for
          preliminary comparison, e.g., labels or properties.

        Returns:
        - Tuple[List[Set[int]], Dict[int, int]]: A tuple containing a list of sets
          (clusters), where each set contains indices of rules in the same cluster,
          and a dictionary mapping each rule index to its cluster index.
        """
        # Determine the appropriate isomorphism function based on rule type
        if isinstance(rules[0], str):
            iso_function = gm._isomorphic_rule
            apply_match_args = (
                False  # rule_isomorphism does not use nodeMatch or edgeMatch
            )
        elif isinstance(rules[0], nx.Graph):
            iso_function = graph_isomorphism
            apply_match_args = True  # graph_isomorphism uses nodeMatch and edgeMatch

        if attributes is None:
            attributes_sorted = [1] * len(rules)
        else:
            if isinstance(attributes[0], str):
                attributes_sorted = attributes
            elif isinstance(attributes, List):
                attributes_sorted = [sorted(value) for value in attributes]
            elif isinstance(attributes, OrderedDict):
                attributes_sorted = [
                    OrderedDict(sorted(value.items())) for value in attributes
                ]

        visited = set()
        clusters = []
        rule_to_cluster = {}

        for i, rule_i in enumerate(rules):
            if i in visited:
                continue
            cluster = {i}
            visited.add(i)
            rule_to_cluster[i] = len(clusters)
            # fmt: off
            for j, rule_j in enumerate(rules[i + 1:], start=i + 1):
                # fmt: on
                if attributes_sorted[i] == attributes_sorted[j] and j not in visited:
                    # Conditionally use matching functions
                    if apply_match_args:
                        # is_isomorphic = iso_function(
                        #     rule_i, rule_j, nodeMatch, edgeMatch
                        # )
                        if aut is not None:
                            for m in aut:
                                _, is_isomorphic =gmap.search_stable_extension(rule_i, rule_j, m, all_extensions=False, node_labels = True, edge_labels = True)
                                if is_isomorphic:
                                    break

                        else:
                            _, is_isomorphic =gmap.search_isomorphisms(rule_i, rule_j, all_isomorphisms=False, node_labels = True, edge_labels = True)
                    else:
                        is_isomorphic = iso_function(rule_i, rule_j)

                    if is_isomorphic:
                        cluster.add(j)
                        visited.add(j)
                        rule_to_cluster[j] = len(clusters)

            clusters.append(cluster)

        return clusters, rule_to_cluster

    def fit(
        self,
        data: List[Dict],
        rule_key: str = "gml",
        attribute_key: str = "WLHash",
        strip: bool = False,
    ) -> List[Dict]:
        """Automatically clusters the rules and assigns them cluster indices
        based on the similarity, potentially using provided templates for
        clustering, or generating new templates.

        Parameters:
        - data (List[Dict]): A list containing dictionaries, each representing a
          rule along with metadata.
        - rule_key (str): The key in the dictionaries under `data` where the rule data
          is stored.
        - attribute_key (str): The key in the dictionaries under `data` where rule
          attributes are stored.

        Returns:
        - List[Dict]: Updated list of dictionaries with an added 'class' key for cluster
          identification.
        """
        if isinstance(data[0][rule_key], str):
            if strip:
                rules = [strip_context(entry[rule_key]) for entry in data]
            else:
                rules = [entry[rule_key] for entry in data]

        else:
            rules = [entry[rule_key] for entry in data]

        attributes = (
            [entry.get(attribute_key) for entry in data] if attribute_key else None
        )
        _, rule_to_cluster_dict = self.iterative_cluster(
            rules, attributes, self.nodeMatch, self.edgeMatch
        )

        for index, entry in enumerate(data):
            entry["class"] = rule_to_cluster_dict.get(index, None)

        return data


class HComplete:
    """A class for infering hydrogen to complete reaction center or ITS
    graph."""

    @staticmethod
    def process_single_graph_data(
        graph_data: Dict[str, nx.Graph],
        its_key: str = "ITS",
        rc_key: str = "RC",
        ignore_aromaticity: bool = False,
        balance_its: bool = True,
        get_priority_graph: bool = False,
        max_hydrogen: int = 7,
    ) -> Dict[str, Optional[nx.Graph]]:
        """Processes a single graph data dictionary by modifying hydrogen
        counts and other features based on configuration settings.

        Parameters:
        - graph_data (Dict[str, nx.Graph]): Dictionary containing the graph data.
        - its_key (str): Key where the ITS graph is stored.
        - rc_key (str): Key where the RC graph is stored.
        - ignore_aromaticity (bool): If True, aromaticity is ignored during processing. Default is False.
        - balance_its (bool): If True, the ITS is balanced. Default is True.
        - get_priority_graph (bool): If True, priority is given to graph data during processing. Default is False.
        - max_hydrogen (int): Maximum number of hydrogens that can be handled in the inference step.

        Returns:
        - Dict[str, Optional[nx.Graph]]: Dictionary with updated ITS and RC graph data, or None if processing fails.
        """
        graphs = copy(graph_data)
        its = graphs.get(its_key, None)
        if not isinstance(its, nx.Graph) or its.number_of_nodes() == 0:
            graphs[its_key], graphs[rc_key] = None, None
            return graphs
        react_graph, prod_graph = its_decompose(its)
        hcount_change = check_hcount_change(react_graph, prod_graph)
        logging.info("HCount change between reactant and product: %d", hcount_change)
        if hcount_change == 0:
            graphs = graphs
        elif hcount_change <= max_hydrogen:
            graphs = HComplete.process_multiple_hydrogens(
                graphs,
                its_key,
                rc_key,
                react_graph,
                prod_graph,
                ignore_aromaticity,
                balance_its,
                get_priority_graph,
            )
        else:
            graphs[its_key], graphs[rc_key] = None, None
        if graphs[rc_key] is not None:
            is_empty_rc_present = (
                not isinstance(graphs[rc_key], nx.Graph)
                or graphs[rc_key].number_of_nodes() == 0
            )

            if is_empty_rc_present:
                graphs[its_key] = None
                graphs[rc_key] = None
        return graphs

    def process_graph_data_parallel(
        self,
        graph_data_list: List[Dict[str, nx.Graph]],
        its_key: str = "ITS",
        rc_key: str = "RC",
        n_jobs: int = 1,
        verbose: int = 0,
        ignore_aromaticity: bool = False,
        balance_its: bool = True,
        get_priority_graph: bool = False,
        max_hydrogen: int = 7,
    ) -> List[Dict[str, Optional[nx.Graph]]]:
        """Processes a list of graph data dictionaries in parallel to optimize
        the hydrogen completion and other graph modifications.

        Parameters:
        - graph_data_list (List[Dict[str, nx.Graph]]): List of dictionaries containing the graph data.
        - its_key (str): Key where the ITS graph is stored.
        - rc_key (str): Key where the RC graph is stored.
        - n_jobs (int): Number of parallel jobs to run.
        - verbose (int): Verbosity level for the parallel process.
        - ignore_aromaticity (bool): If True, aromaticity is ignored during processing. Default is False.
        - balance_its (bool): If True, the ITS is balanced. Default is True.
        - get_priority_graph (bool): If True, priority is given to graph data during processing. Default is False.
        - max_hydrogen (int): Maximum number of hydrogens that can be handled in the inference step.

        Returns:
        - List[Dict[str, Optional[nx.Graph]]]: List of dictionaries with
        updated ITS and RC graph data, or None if processing fails.
        """
        processed_data = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(self.process_single_graph_data)(
                graph_data,
                its_key,
                rc_key,
                ignore_aromaticity,
                balance_its,
                get_priority_graph,
                max_hydrogen,
            )
            for graph_data in graph_data_list
        )

        return processed_data

    @staticmethod
    def process_multiple_hydrogens(
        graph_data: Dict[str, nx.Graph],
        its_key: str,
        rc_key: str,
        react_graph: nx.Graph,
        prod_graph: nx.Graph,
        ignore_aromaticity: bool,
        balance_its: bool,
        get_priority_graph: bool = False,
    ) -> Dict[str, Optional[nx.Graph]]:
        """Handles significant hydrogen count changes between reactant and
        product graphs, adjusting hydrogen nodes accordingly and assessing
        graph equivalence.

        Parameters:
        - graph_data (Dict[str, nx.Graph]): Dictionary containing the graph data.
        - its_key (str): Key for the ITS graph in the dictionary.
        - rc_key (str): Key for the RC graph in the dictionary.
        - react_graph (nx.Graph): Graph representing the reactants.
        - prod_graph (nx.Graph): Graph representing the products.
        - ignore_aromaticity (bool): If True, aromaticity will not be considered in processing.
        - balance_its (bool): If True, balances the ITS graph.
        - get_priority_graph (bool): If True, processes graphs with priority considerations.

        Returns:
        - Dict[str, Optional[nx.Graph]]: Updated graph dictionary with potentially modified ITS and RC graphs.
        """
        combinations_solution = HComplete.add_hydrogen_nodes_multiple(
            react_graph,
            prod_graph,
            ignore_aromaticity,
            balance_its,
            get_priority_graph,
        )
        if len(combinations_solution) == 0:
            graph_data[its_key], graph_data[rc_key] = None, None
            return graph_data

        filtered_combinations_solution = []
        react_list = []
        prod_list = []
        rc_list = []
        its_list = []
        rc_sig = []

        for react, prod, its, rc, sig in combinations_solution:
            if rc is not None and isinstance(rc, nx.Graph) and rc.number_of_nodes() > 0:
                filtered_combinations_solution.append((react, prod, rc, its, sig))
                react_list.append(react)
                prod_list.append(prod)
                rc_list.append(rc)
                its_list.append(its)
                rc_sig.append(sig)

        if len(set(rc_sig)) != 1:
            equivariant = 0
        else:
            _, equivariant = check_equivariant_graph(rc_list)

        pairwise_combinations = len(rc_list) - 1
        if equivariant == pairwise_combinations:
            graph_data[its_key] = its_list[0]
            graph_data[rc_key] = rc_list[0]
        else:
            graph_data[its_key], graph_data[rc_key] = None, None
            if get_priority_graph:
                priority_indices = get_priority(rc_list)
                rc_list = [rc_list[i] for i in priority_indices]
                rc_sig = [rc_sig[i] for i in priority_indices]
                its_list = [its_list[i] for i in priority_indices]
                react_list = [react_list[i] for i in priority_indices]
                prod_list = [prod_list[i] for i in priority_indices]
                if len(set(rc_sig)) == 1:
                    _, equivariant = check_equivariant_graph(rc_list)
                pairwise_combinations = len(rc_list) - 1
                if equivariant == pairwise_combinations:
                    graph_data[its_key] = its_list[0]
                    graph_data[rc_key] = rc_list[0]
        return graph_data

    @staticmethod
    def add_hydrogen_nodes_multiple(
        react_graph: nx.Graph,
        prod_graph: nx.Graph,
        ignore_aromaticity: bool,
        balance_its: bool,
        get_priority_graph: bool = False,
    ) -> List[Tuple[nx.Graph, nx.Graph]]:
        """Generates multiple permutations of reactant and product graphs by
        adjusting hydrogen counts, exploring all possible configurations of
        hydrogen node additions or removals.

        Parameters:
        - react_graph (nx.Graph): The reactant graph.
        - prod_graph (nx.Graph): The product graph.
        - ignore_aromaticity (bool): If True, aromaticity is ignored.
        - balance_its (bool): If True, attempts to balance the ITS by adjusting hydrogen nodes.
        - get_priority_graph (bool): If True, additional priority-based processing
        is applied to select optimal graph configurations.

        Returns:
        - List[Tuple[nx.Graph, nx.Graph]]: A list of graph tuples, each representing
        a possible configuration of reactant and product graphs with adjusted hydrogen nodes.
        """
        react_graph_copy = react_graph.copy()
        prod_graph_copy = prod_graph.copy()
        react_explicit_h, hydrogen_nodes = check_explicit_hydrogen(react_graph_copy)
        prod_explicit_h, _ = check_explicit_hydrogen(prod_graph_copy)
        hydrogen_nodes_form, hydrogen_nodes_break = [], []

        primary_graph = (
            react_graph_copy if react_explicit_h <= prod_explicit_h else prod_graph_copy
        )
        for node_id in primary_graph.nodes:
            try:
                # Calculate the difference in hydrogen counts
                hcount_diff = react_graph_copy.nodes[node_id].get(
                    "hcount", 0
                ) - prod_graph_copy.nodes[node_id].get("hcount", 0)
            except KeyError:
                # Handle cases where node_id does not exist in opposite_graph
                continue

            # Decide action based on hcount_diff
            if hcount_diff > 0:
                hydrogen_nodes_break.extend([node_id] * hcount_diff)
            elif hcount_diff < 0:
                hydrogen_nodes_form.extend([node_id] * -hcount_diff)

        max_index = max(
            max(react_graph_copy.nodes, default=0),
            max(prod_graph_copy.nodes, default=0),
        )
        range_implicit_h = range(
            max_index + 1,
            max_index + 1 + len(hydrogen_nodes_form) - react_explicit_h,
        )
        combined_indices = list(range_implicit_h) + hydrogen_nodes
        permutations = list(itertools.permutations(combined_indices))
        permutations_seed = permutations[0]

        updated_graphs = []
        for permutation in permutations:
            current_react_graph, current_prod_graph = react_graph_copy, prod_graph_copy

            new_hydrogen_node_ids = [i for i in permutations_seed]

            # Use `zip` to pair `hydrogen_nodes_break` with the new IDs
            node_id_pairs = zip(hydrogen_nodes_break, new_hydrogen_node_ids)
            # Call the method with the formed pairs and specify atom_map_update as False
            current_react_graph = HComplete.add_hydrogen_nodes_multiple_utils(
                current_react_graph, node_id_pairs, atom_map_update=False
            )
            # Varied hydrogen nodes in the product graph based on permutation
            current_prod_graph = HComplete.add_hydrogen_nodes_multiple_utils(
                current_prod_graph, zip(hydrogen_nodes_form, permutation)
            )
            its = ITSConstruction().ITSGraph(
                current_react_graph,
                current_prod_graph,
                ignore_aromaticity=ignore_aromaticity,
                balance_its=balance_its,
            )
            rc = get_rc(its)
            sig = WLHash(iterations=3).weisfeiler_lehman_graph_hash(rc)
            if get_priority_graph is False:
                if len(updated_graphs) > 0:
                    if sig != updated_graphs[-1][-1]:
                        return []
            updated_graphs.append(
                (current_react_graph, current_prod_graph, its, rc, sig)
            )
        return updated_graphs

    @staticmethod
    def add_hydrogen_nodes_multiple_utils(
        graph: nx.Graph,
        node_id_pairs: Iterable[Tuple[int, int]],
        atom_map_update: bool = True,
    ) -> nx.Graph:
        """Creates and returns a new graph with added hydrogen nodes based on
        the input graph and node ID pairs.

        Parameters:
        - graph (nx.Graph): The base graph to which the nodes will be added.
        - node_id_pairs (Iterable[Tuple[int, int]]): Pairs of node IDs (original node, new
        hydrogen node) to link with hydrogen.
        - atom_map_update (bool): If True, update the 'atom_map' attribute with the new
        hydrogen node ID; otherwise, retain the original node's 'atom_map'.

        Returns:
        - nx.Graph: A new graph instance with the added hydrogen nodes.
        """
        new_graph = deepcopy(graph)
        for node_id, new_hydrogen_node_id in node_id_pairs:
            atom_map_val = (
                new_hydrogen_node_id
                if atom_map_update
                else new_graph.nodes[node_id].get("atom_map", 0)
            )
            new_graph.add_node(
                new_hydrogen_node_id,
                charge=0,
                hcount=0,
                aromatic=False,
                element="H",
                atom_map=atom_map_val,
                # isomer="N",
                # partial_charge=0,
                # hybridization=0,
                # in_ring=False,
                # explicit_valence=0,
                # implicit_hcount=0,
            )
            new_graph.add_edge(
                node_id,
                new_hydrogen_node_id,
                order=1.0,
                # ez_isomer="N",
                bond_type="SINGLE",
                # conjugated=False,
                # in_ring=False,
            )
            new_graph.nodes[node_id]["hcount"] -= 1
        return new_graph


import gmapache as gmap

cluster = GraphCluster()


class HExtend(HComplete):

    @staticmethod
    def get_unique_graphs_for_clusters(
        graphs: List[nx.Graph], cluster_indices: List[set]
    ) -> List[nx.Graph]:
        """Retrieve a unique graph for each cluster from a list of graphs based
        on cluster indices.

        This method selects one graph per cluster based on the first index found
        in each cluster set. Note: Clusters are expected to be represented
        as sets of indices, each corresponding to a graph in the `graphs` list.

        Parameters:
        - graphs (List[nx.Graph]): List of networkx graphs.
        - cluster_indices (List[set]): List of sets, each containing indices representing graphs
        that belong to the same cluster.

        Returns:
        - List[nx.Graph]: A list containing one unique graph from each cluster. The graph chosen
        is the one corresponding to the first index in each cluster set, which is arbitrary
        due to the unordered nature of sets.

        Raises:
        - ValueError: If any index in `cluster_indices` is out of the range of `graphs`.
        - TypeError: If `cluster_indices` is not a list of sets.
        """
        if not all(isinstance(cluster, set) for cluster in cluster_indices):
            raise TypeError("Each cluster index must be a set of integers.")
        if any(
            min(cluster) < 0 or max(cluster) >= len(graphs)
            for cluster in cluster_indices
            if cluster
        ):
            raise ValueError("Cluster indices are out of the range of the graphs list.")

        unique_graphs = [
            graphs[next(iter(cluster))] for cluster in cluster_indices if cluster
        ]
        return unique_graphs

    @staticmethod
    def _extend(
        its: nx.Graph,
        ignore_aromaticity: bool,
        balance_its: bool,
    ) -> Tuple[List[nx.Graph], List[nx.Graph], List[str]]:
        """Process equivalent maps by adding hydrogen nodes and constructing
        ITS graphs based on the balance and aromaticity settings.

        Parameters:
        - its (nx.Graph): The initial transition state graph to be processed.
        - ignore_aromaticity (bool): Flag to ignore aromaticity in graph construction.
        - balance_its (bool): Flag to balance the ITS graph during processing.

        Returns:
        - Tuple[List[nx.Graph], List[nx.Graph], List[str]]: Tuple containing lists of
        processed reaction graphs, ITS graphs, and their signatures.
        """
        react_graph, prod_graph = its_decompose(its)
        hcount_change = check_hcount_change(react_graph, prod_graph)
        if hcount_change == 0:
            its_list = [its]
            rc_list = [get_rc(its)]
            # sigs = [
            #     WLHash(iterations=3).weisfeiler_lehman_graph_hash(i) for i in rc_list
            # ]
            sigs = []
            return rc_list, its_list, sigs

        combinations_solution = HComplete.add_hydrogen_nodes_multiple(
            react_graph,
            prod_graph,
            ignore_aromaticity,
            balance_its,
            get_priority_graph=True,
        )

        rc_list, its_list, rc_sig = [], [], []
        for _, _, its, rc, sig in combinations_solution:
            if rc and isinstance(rc, nx.Graph) and rc.number_of_nodes() > 0:
                rc_list.append(rc)
                its_list.append(its)
                rc_sig.append(sig)
        return rc_list, its_list, rc_sig

    @staticmethod
    def _process(
        data_dict: Dict,
        its_key: str,
        rc_key: str,
        ignore_aromaticity: bool,
        balance_its: bool,
        use_aut: bool = False,
    ) -> Dict:
        """Processes a dictionary of graphs using specific graph processing
        functions and updates the dictionary with new graph data.

        Parameters:
        - data_dict (Dict): Dictionary containing the graphs and their keys.
        - its_key (str): Key in the dictionary for the ITS graph.
        - rc_key (str): Key in the dictionary for the reaction graph.
        - ignore_aromaticity (bool): Whether to ignore aromaticity
        during graph processing.
        - balance_its (bool): Whether to balance the ITS graph.

        Returns:
        - Dict: The updated dictionary containing new ITS and reaction graphs.
        """
        its = data_dict[its_key]
        aut = None
        if use_aut:
            aut, _ = gmap.search_isomorphisms(
                data_dict[rc_key],
                data_dict[rc_key],
                all_isomorphisms=True,
                node_labels=True,
                edge_labels=True,
            )
            logging.info(f"Number of automorphisms found: {len(aut)}")

        rc_list, its_list, rc_sig = HExtend._extend(
            its, ignore_aromaticity, balance_its
        )
        rc_sig = None
        cls, _ = cluster.iterative_cluster(rc_list, rc_sig, aut=aut)
        new_rc = HExtend.get_unique_graphs_for_clusters(rc_list, cls)
        new_its = HExtend.get_unique_graphs_for_clusters(its_list, cls)
        data_dict[rc_key] = new_rc
        data_dict[its_key] = new_its
        return data_dict

    @staticmethod
    def fit(
        data,
        its_key: str,
        rc_key: str,
        ignore_aromaticity: bool = False,
        balance_its: bool = True,
        use_aut: bool = False,
    ) -> List:
        """Fit the model to the data in parallel, processing each entry to
        generate new graph data based on the ITS and reaction graph keys.

        Parameters:
        - data (iterable): Data to be processed.
        - its_key (str): Key for the ITS graphs in the data.
        - rc_key (str): Key for the reaction graphs in the data.
        - ignore_aromaticity (bool): Whether to ignore aromaticity during processing.
        Default to False.
        - balance_its (bool): Whether to balance the ITS during processing.
        Default to True.
        - n_jobs (int): Number of jobs to run in parallel. Default to 1.
        - verbose (int): Verbosity level for parallel processing. Default to 0.

        Returns:
        - List: A list containing the results of the processed data.
        """
        results = []
        for key, item in enumerate(data):
            try:
                result = HExtend._process(
                    item, its_key, rc_key, ignore_aromaticity, balance_its, use_aut
                )
                results.append(result)
            except Exception as e:
                logging.error(f"Error processing item at index {key}: {e}")
                results.append(item)
        return results
