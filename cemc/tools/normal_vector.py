from scipy.spatial import cKDTree as KDTree
import numpy as np

class NormalVectorEstimator(object):
    def __init__(self, simplices, points):
        self.simplices = simplices
        self.points = points
        self.centroids = self._facet_centroids()
        self.centroid_tree = KDTree(self.centroids)
        self.normal_vectors = self._facet_normals()
        self.num_average_history = []

    def reset(self):
        """Reset the normal vector estimator."""
        self.num_average_history = []

    def _facet_centroids(self):
        """Calculate the centroids of all facets."""
        centroids = np.zeros((len(self.simplices), 3))
        for i, facet in enumerate(self.simplices):
            v1 = self.points[facet[0], :]
            v2 = self.points[facet[1], :]
            v3 = self.points[facet[2], :]
            centroids[i, :] = (v1 + v2 + v3)/3
        return centroids
    
    def _facet_normals(self):
        """Calculate the facet normals."""
        com = np.mean(self.points, axis=0)
        normals = np.zeros((len(self.simplices), 3))
        for i, facet in enumerate(self.simplices):
            v1 = self.points[facet[0], :]
            v2 = self.points[facet[1], :]
            v3 = self.points[facet[2], :]
            
            vec1 = v2 - v1
            vec2 = v3 - v1
            n = np.cross(vec1, vec2)
            n /= np.sqrt(n.dot(n))
            orig_to_centroid = self.centroids[i, :] - com

            # We have to make sure that we store the outwards normal vector
            dist = orig_to_centroid.dot(n)
            if dist < 0.0:
                normals[i, :] = -n
            else:
                normals[i, :] = n
        return normals

    def get_normal(self, x, cutoff=1.0):
        """Calculate the normal vector at point on the surface.
        
        :param np.ndarray x: Position at which to calculate the normal
        :param float cutoff: Local averaging cutoff. The returned 
            vector is the mean of the normal of all facets whose
            centroid is less that this value away from x.
        """
        facet_indx = self.centroid_tree.query_ball_point(x, cutoff)
        if not facet_indx:
            raise ValueError("There are now facets with {} from {}"
                             "Are you sure you specified a point on the " 
                             "surface?".format(cutoff, x))
        
        normal = np.zeros(3)
        for indx in facet_indx:
            normal += self.normal_vectors[indx, :]
        self.num_average_history.append(len(facet_indx))
        return normal/np.sqrt(normal.dot(normal))

    def show_statistics(self):
        """Plots the statistics of averaging."""
        from matplotlib import pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(self.num_average_history, "o", mfc="none")
        ax.set_xlabel("Call number")
        ax.set_ylabel("Number of facet used in averaging")
        plt.show()