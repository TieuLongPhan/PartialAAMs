import gmapache as gm
from partialaams.utils import (
    get_aam_pairwise_indices,
    create_adjacency_matrix,
    get_list_of_rsmi,
)


def gm_extend_from_graph(G, H):
    M = get_aam_pairwise_indices(G, H)
    all_extensions, _ = gm.maximum_connected_extensions(G, H, M)
    Ms = [create_adjacency_matrix(value) for value in all_extensions]
    return get_list_of_rsmi(G, H, Ms)
