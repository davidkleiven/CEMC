

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
