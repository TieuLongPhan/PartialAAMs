import networkx as nx
from rdkit import Chem
from pathlib import Path
from aamutils.utils import smiles_to_graph, graph_to_mol
from aamutils.algorithm.aaming import get_its, get_rc
from synutility.SynIO.data_type import load_database, save_to_pickle


def get_partial_aam_rc(rsmi):
    G, H = smiles_to_graph(rsmi)
    if len(G.nodes) != len(H.nodes):
        return None, None

    ITS = get_its(G, H)
    RC = get_rc(ITS)

    rc_nodes = RC.nodes
    nodes = list(ITS.nodes)
    nodes = [n for n in nodes if n not in rc_nodes]

    for rand_n in nodes:
        G_idx, H_idx = nx.get_node_attributes(ITS, "idx_map")[rand_n]
        G.nodes[G_idx]["aam"] = 0
        H.nodes[H_idx]["aam"] = 0

    return G, H


def graphs_to_smiles(r, p):
    reactants = graph_to_mol(r)
    products = graph_to_mol(p)
    return f"{Chem.MolToSmiles(reactants)}>>{Chem.MolToSmiles(products)}"


def data_prepare(input_file, output_file):
    data = load_database(input_file)
    data = [value for value in data if value["equivalent"]]

    for value in data:
        rsmi = value["local_mapper"]
        results = get_partial_aam_rc(rsmi)
        value["G"], value["H"] = results[0], results[1]

    data = [value for value in data if value["G"]]
    for value in data:
        value["partial_aam_rsmi"] = graphs_to_smiles(value["G"], value["H"])

    if output_file:
        save_to_pickle(data, output_file)


if __name__ == "__main__":
    root_dir = Path(__file__).parents[1]
    data_prepare(
        f"{root_dir}/Data/test_dataset.json",
        f"{root_dir}/Data/test_dataset_partial_aam.pkl.gz",
    )
