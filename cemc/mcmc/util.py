from ase.units import kB
import numpy as np


def unique_cluster_indices(cluster_indx):
    """
    Return a unique list of cluster indices

    :param cluster_indx: Cluster index list of ClusterExpansionSetting
    """
    unique_indx = []
    for symmgroup in cluster_indx:
        for sizegroup in symmgroup:
            for cluster in sizegroup:
                if cluster is None:
                    continue
                for subcluster in cluster:
                    for indx in subcluster:
                        if indx not in unique_indx:
                            unique_indx.append(indx)
    return unique_indx


def trans_matrix2listdict(BC):
    """
    Converts the translation matrix to a list of dictionaries

    :param BC: Instance of the bulk crystal
    """
    matrix = BC.trans_matrix
    unique_indx = unique_cluster_indices(BC.cluster_indx)
    print(unique_indx)
    listdict = []
    for refindx in range(matrix.shape[0]):
        listdict.append({})
        for indx in unique_indx:
            listdict[refindx][indx] = matrix[refindx, indx]
    return listdict


def waste_recycled_average(observable, energies, T):
    """Compute average by using all energy values."""
    E0 = np.min(energies)
    dE = energies - E0
    beta = 1.0/(kB*T)
    w = np.exp(-beta*dE)
    return observable.dot(w)/np.sum(w)


def waste_recycled_accept_prob(energies, T):
    """Compute acceptance probability in the weight recycled scheme."""
    E0 = np.min(energies)
    dE = energies - E0
    beta = 1.0/(kB*T)
    w = np.exp(-beta * dE)
    return w/np.sum(w)


def get_new_state(weights):
    """Select a new state according to the weights."""
    srt_indx = np.argsort(weights)
    srt_weights = np.sort(weights)
    rand_num = np.random.rand()
    indx = np.searchsorted(srt_weights, rand_num)
    return srt_indx[indx]
