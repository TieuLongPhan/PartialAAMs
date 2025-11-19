import io
import time
import logging
from synkit.IO import load_from_pickle, save_to_pickle

import networkx as nx
from typing import List, Tuple, Dict

from synkit.Graph.Hyrogen._misc import check_hcount_change
from synkit.Graph.ITS.its_decompose import get_rc, its_decompose

import itertools
from copy import deepcopy, copy
from joblib import Parallel, delayed
from typing import Dict, List, Tuple, Iterable, Optional

from synkit.Graph.Feature.wl_hash import WLHash
from synkit.Graph.ITS.its_construction import ITSConstruction
from synkit.Graph.ITS.its_decompose import get_rc, its_decompose
from synkit.Graph.Hyrogen._misc import (
    check_hcount_change,
    check_explicit_hydrogen,
    get_priority,
    check_equivariant_graph,
)

import importlib.util
from operator import eq
from collections import OrderedDict
from typing import List, Set, Dict, Any, Tuple, Optional, Callable
from networkx.algorithms.isomorphism import generic_node_match, generic_edge_match

from synkit.Rule.Modify.rule_utils import strip_context
from synkit.Graph.Matcher.graph_morphism import graph_isomorphism
from synkit.Graph.Matcher.graph_matcher import GraphMatcherEngine
import gmapache as gmap

logger = logging.getLogger(__name__)

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
        import importlib.util

        backends = ["nx"]
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
        if isinstance(rules[0], str):
            iso_function = gm._isomorphic_rule
            apply_match_args = False
        elif isinstance(rules[0], nx.Graph):
            iso_function = graph_isomorphism
            apply_match_args = True

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
            for j, rule_j in enumerate(rules[i + 1 :], start=i + 1):
                if attributes_sorted[i] == attributes_sorted[j] and j not in visited:
                    if apply_match_args:
                        if aut is not None:
                            is_isomorphic = False
                            for m in aut:
                                _, is_isomorphic = gmap.search_stable_extension(
                                    rule_i,
                                    rule_j,
                                    m,
                                    all_extensions=False,
                                    node_labels=True,
                                    edge_labels=True,
                                )
                                if is_isomorphic:
                                    break
                        else:
                            _, is_isomorphic = gmap.search_isomorphisms(
                                rule_i,
                                rule_j,
                                all_isomorphisms=False,
                                node_labels=True,
                                edge_labels=True,
                            )
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
        graphs = copy(graph_data)
        its = graphs.get(its_key, None)
        if not isinstance(its, nx.Graph) or its.number_of_nodes() == 0:
            graphs[its_key], graphs[rc_key] = None, None
            return graphs
        react_graph, prod_graph = its_decompose(its)
        hcount_change = check_hcount_change(react_graph, prod_graph)
        logger.info("HCount change between reactant and product: %d", hcount_change)
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
        combinations_solution = HComplete.add_hydrogen_nodes_multiple(
            react_graph,
            prod_graph,
            ignore_aromaticity,
            balance_its,
            get_priority_graph,
        )
        logger.info("Number of H-combination solutions: %d", len(combinations_solution))
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
        logger.info(
            "Equivariant RC count: %d / %d", equivariant, max(pairwise_combinations, 0)
        )
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
                logger.info(
                    "After priority, equivariant RC count: %d / %d",
                    equivariant,
                    max(pairwise_combinations, 0),
                )
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
        react_graph_copy = react_graph.copy()
        prod_graph_copy = prod_graph.copy()
        react_explicit_h, hydrogen_nodes = check_explicit_hydrogen(react_graph_copy)
        prod_explicit_h, _ = check_explicit_hydrogen(prod_graph_copy)
        hydrogen_nodes_form, hydrogen_nodes_break = [], []

        logger.info("Explicit H: react=%d prod=%d", react_explicit_h, prod_explicit_h)

        primary_graph = (
            react_graph_copy if react_explicit_h <= prod_explicit_h else prod_graph_copy
        )
        for node_id in primary_graph.nodes:
            try:
                hcount_diff = react_graph_copy.nodes[node_id].get(
                    "hcount", 0
                ) - prod_graph_copy.nodes[node_id].get("hcount", 0)
            except KeyError:
                continue

            if hcount_diff > 0:
                hydrogen_nodes_break.extend([node_id] * hcount_diff)
            elif hcount_diff < 0:
                hydrogen_nodes_form.extend([node_id] * -hcount_diff)

        logger.info(
            "Hydrogen to break: %d, to form: %d",
            len(hydrogen_nodes_break),
            len(hydrogen_nodes_form),
        )

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
        logger.info("Total implicit H permutations: %d", len(permutations))
        if not permutations:
            return []

        permutations_seed = permutations[0]

        updated_graphs = []
        for permutation in permutations:
            current_react_graph, current_prod_graph = react_graph_copy, prod_graph_copy

            new_hydrogen_node_ids = [i for i in permutations_seed]

            node_id_pairs = zip(hydrogen_nodes_break, new_hydrogen_node_ids)
            current_react_graph = HComplete.add_hydrogen_nodes_multiple_utils(
                current_react_graph, node_id_pairs, atom_map_update=False
            )
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
            sig = None
            # sig = WLHash(iterations=3).weisfeiler_lehman_graph_hash(rc)
            # if get_priority_graph is False:
            #     if len(updated_graphs) > 0:
            #         if sig != updated_graphs[-1][-1]:
            #             return []
            updated_graphs.append(
                (current_react_graph, current_prod_graph, its, rc, sig)
            )
        logger.info("Updated graph candidates: %d", len(updated_graphs))
        return updated_graphs

    @staticmethod
    def add_hydrogen_nodes_multiple_utils(
        graph: nx.Graph,
        node_id_pairs: Iterable[Tuple[int, int]],
        atom_map_update: bool = True,
    ) -> nx.Graph:
        new_graph = deepcopy(graph)
        for node_id, new_hydrogen_node_id in node_id_pairs:
            atom_map_val = (
                new_hydrogen_node_id
                if atom_map_update
                else new_graph.nodes[node_id].get("atom_map", 0)
            )
            logger.debug(
                "Adding H node %d attached to %d (atom_map=%d)",
                new_hydrogen_node_id,
                node_id,
                atom_map_val,
            )
            new_graph.add_node(
                new_hydrogen_node_id,
                charge=0,
                hcount=0,
                aromatic=False,
                element="H",
                atom_map=atom_map_val,
            )
            new_graph.add_edge(
                node_id,
                new_hydrogen_node_id,
                order=1.0,
                bond_type="SINGLE",
            )
            new_graph.nodes[node_id]["hcount"] -= 1
        return new_graph


cluster = GraphCluster()


class HExtend(HComplete):
    @staticmethod
    def get_unique_graphs_for_clusters(
        graphs: List[nx.Graph], cluster_indices: List[set]
    ) -> List[nx.Graph]:
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
        react_graph, prod_graph = its_decompose(its)
        hcount_change = check_hcount_change(react_graph, prod_graph)
        logger.info("HExtend ΔH between reactant and product: %d", hcount_change)
        if hcount_change == 0:
            its_list = [its]
            rc_list = [get_rc(its)]
            sigs = []
            logger.info("HExtend: no H change, single RC/ITS kept")
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
        logger.info("HExtend: RC candidates after H-extend: %d", len(rc_list))
        return rc_list, its_list, rc_sig

    @staticmethod
    def _process(
        data_dict: Dict,
        its_key: str,
        rc_key: str,
        ignore_aromaticity: bool,
        balance_its: bool,
        use_aut: bool = False,
        use_rc: bool = False,
    ) -> Dict:
        its = data_dict[its_key]
        aut = None
        if use_aut:
            if use_rc:

                aut, _ = gmap.search_isomorphisms(
                    data_dict[rc_key],
                    data_dict[rc_key],
                    all_isomorphisms=True,
                    node_labels=True,
                    edge_labels=True,
                )

            else:
                aut, _ = gmap.search_isomorphisms(
                    data_dict[its_key],
                    data_dict[its_key],
                    all_isomorphisms=True,
                    node_labels=True,
                    edge_labels=True,
                )
            logger.info("Number of automorphisms found: %d", len(aut))

        rc_list, its_list, rc_sig = HExtend._extend(
            its, ignore_aromaticity, balance_its
        )
        rc_sig = None
        if use_rc:

            cls, _ = cluster.iterative_cluster(rc_list, rc_sig, aut=aut)
        else:
            cls, _ = cluster.iterative_cluster(its_list, rc_sig, aut=aut)
        logger.info("Number of RC clusters: %d", len(cls))
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
        use_rc: bool = False,
    ) -> List:
        results = []
        for key, item in enumerate(data):
            try:
                result = HExtend._process(
                    item,
                    its_key,
                    rc_key,
                    ignore_aromaticity,
                    balance_its,
                    use_aut,
                    use_rc,
                )
                results.append(result)
            except Exception as e:
                logger.error("Error processing item at index %d: %s", key, e)
                results.append(item)
        return results


if __name__ == "__main__":
    from copy import deepcopy

    def drop_atom_map(G: nx.Graph, inplace: bool = False) -> nx.Graph:
        """
        Remove 'atom_map' attribute from all nodes in a graph.

        Parameters
        ----------
        G : nx.Graph
            Input graph.
        inplace : bool, default False
            If False, return a new graph. If True, modify `G` directly.

        Returns
        -------
        nx.Graph
            The graph with 'atom_map' attributes removed.
        """
        H = G if inplace else deepcopy(G)

        for n, data in H.nodes(data=True):
            if "atom_map" in data:
                data.pop("atom_map", None)

        return H

    DATA_PICKLE_IN = "./Data/hydrogen.pkl.gz"
    DATA_PICKLE_OUT = "./Data/hydrogen_hextend_enriched.pkl.gz"
    MASTER_LOG = "./Data/hextend_run.log"

    # ------------------------------------------------------------
    # Logging setup
    # ------------------------------------------------------------
    fmt = "%(asctime)s %(levelname)s %(name)s %(message)s"
    formatter = logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    for h in list(root.handlers):
        root.removeHandler(h)

    fh = logging.FileHandler(MASTER_LOG, mode="a")
    fh.setFormatter(formatter)
    root.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    root.addHandler(ch)

    logger_main = logging.getLogger("hextend_loop")

    # ------------------------------------------------------------
    # Load input
    # ------------------------------------------------------------
    original_data = load_from_pickle(DATA_PICKLE_IN)
    start_idx = 0
    stop_idx = len(original_data)
    for v in original_data:
        v["ITS"] = drop_atom_map(v["ITS"])
        v["RC"] = drop_atom_map(v["RC"])

    # ------------------------------------------------------------
    # Define scenarios
    # ------------------------------------------------------------
    scenarios = [
        {"use_aut": False, "use_rc": False},
        {"use_aut": True, "use_rc": False},
        {"use_aut": False, "use_rc": True},
        {"use_aut": True, "use_rc": True},
    ]

    # ------------------------------------------------------------
    # Helper function for processing one entry under one scenario
    # ------------------------------------------------------------
    def run_single(entry, idx, scenario):
        item_id = entry.get("R-id", f"idx-{idx}")
        label = f"AUT:{scenario['use_aut']}_RC:{scenario['use_rc']}"

        logger_main.info(f"--- START ({label}) for {item_id} (index {idx}) ---")

        sio = io.StringIO()
        handler = logging.StreamHandler(sio)
        handler.setFormatter(formatter)
        root.addHandler(handler)

        t0 = time.perf_counter()
        out, error = None, None

        try:
            res = HExtend().fit(
                [entry],
                "ITS",
                "RC",
                use_aut=scenario["use_aut"],
                use_rc=scenario["use_rc"],
            )
            out = res[0] if isinstance(res, (list, tuple)) and len(res) == 1 else res

            elapsed = time.perf_counter() - t0
            logger_main.info(f"({label}) Finished {item_id} in {elapsed:.3f}s")

        except Exception as exc:
            elapsed = time.perf_counter() - t0
            error = str(exc)
            logger_main.exception(
                f"({label}) FAILED for {item_id} (index {idx}) after {elapsed:.3f}s"
            )

        finally:
            root.removeHandler(handler)
            log_text = sio.getvalue()
            sio.close()
            for h in root.handlers:
                try:
                    h.flush()
                except Exception:
                    pass

        return out, elapsed, log_text, error

    # ------------------------------------------------------------
    # Main execution loop
    # ------------------------------------------------------------
    for scenario in scenarios:
        data = deepcopy(original_data)
        label = f"AUT{scenario['use_aut']}_RC{scenario['use_rc']}"
        logger_main.info(f"=== START SCENARIO {label} ===")

        scenario_results = []
        for idx in range(start_idx, stop_idx):
            out, elapsed, log_text, error = run_single(data[idx], idx, scenario)

            key_prefix = f"HEXTEND_{label}"

            data[idx][f"{key_prefix}_result"] = out
            data[idx][f"{key_prefix}_time_s"] = elapsed
            data[idx][f"{key_prefix}_log"] = log_text
            if error:
                data[idx][f"{key_prefix}_error"] = error

            scenario_results.append(elapsed)

            save_to_pickle(data, DATA_PICKLE_OUT)

        total_time = sum(scenario_results)
        logger_main.info(
            f"=== FINISHED SCENARIO {label}; total entries {len(scenario_results)}; total time {total_time:.3f}s ==="
        )
