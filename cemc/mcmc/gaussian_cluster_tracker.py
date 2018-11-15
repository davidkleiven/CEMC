import numpy as np
from itertools import combinations_with_replacement

class CouldNotClassifyClusterError(Exception):
    pass

class GaussianClusterTracker(object):
    def __init__(self, atoms=None, threshold=0.001, 
                 cluster_elements=[], max_num_clusters=100):
        self.atoms = atoms
        self._nn_dist= self._nn_distance()
        self.threshold = threshold
        self.cluster_id = -np.ones(len(self.atoms), dtype=np.int8)
        self.probability = np.zeros(len(self.atoms))
        self.gaussians = []
        self.num_members = []
        self.cluster_elements = cluster_elements
        self._next_cluster_id = 0
        self.max_num_clusters = max_num_clusters

    def add_gaussian(self, mu, sigma=None):
        from cemc.tools import MultivariateGaussian
        if sigma is None:
            sigma = np.diag(self._nn_dist**2, 3)
        cell = self.atoms.get_cell()
        self.gaussians.append(
            MultivariateGaussian(cell=cell, mu=mu, sigma=sigma))
        self.num_members.append(0)
        self._next_cluster_id += 1
        return self._next_cluster_id - 1

    @property
    def num_clusters(self):
        return len(self.gaussians)

    def likely_hoods(self, x):
        """Compute all the likely hoods for all distribution."""
        prob = []
        for gauss in self.gaussians:
            prob.append(gauss(x))
        return prob/np.sum(prob)

    def move_create_new_cluster(self, system_changes):
        """Check if the proposed move create a new cluster."""
        for change in system_changes:
            if change[2] in self.cluster_elements:
                prob = self.likely_hoods(self.atoms[change[0]].positions)
                if np.max(prob) < self.threshold:
                    return True
        return False

    def update_clusters(self, system_changes):
        """Classify and update the cluster values."""
        for change in system_changes:
            if change[1] in self.cluster_elements and change[2] not in self.cluster_elements:
                # One atom is removed
                x = self.atoms[change[0]].position
                old_uid = self.cluster_id[change[0]]
                mu = self.gaussians[old_uid].mu
                sigma1 = self.gaussians[old_uid].sigma

                mu2 = mu
                sigma2 = sigma1
                N = self.num_members[old_uid]
                mu1_mu1T = np.outer(mu, mu)
                mu2_mu2T = np.outer(mu2, mu2)
                
                if N > 1:
                    mu2 = (N*mu - x)/(N-1)
                    sigma2 = (N*sigma1 + mu1_mu1T - np.outer(x, x) - mu2_mu2T)/(N-1)

                self.num_members[old_uid] = N-1
                self.gaussians[old_uid].mu = mu2
                self.gaussians[old_uid].sigma = sigma2
                self.cluster_id[old_uid] = -1
            elif change[1] not in self.cluster_elements and change[2] in self.cluster_elements:
                prob = self.likely_hoods(self.atoms[change[0]].position)
                uid = np.argmax(prob)
                self.cluster_id[change[0]] = uid
                N = self.num_members[uid]
                mu = self.gaussians[uid].mu
                sigma1 = self.gaussians[old_uid].sigma
                mu2 = (N*mu + self.atoms[change[0]].position)/(N+1)
                mu1_mu1T = np.outer(mu, mu)
                mu2_mu2T = np.outer(mu2, mu2)
                sigma2 = (N*sigma1 + mu1_mu1T + np.outer(x, x) - mu2_mu2T)/(N+1)
                self.gaussians[uid].mu = mu2
                self.gaussians[uid].sigma = sigma2
                self.num_members[uid] = N+1

    def _nn_distance(self):
        """Find the nearest neighbour distance."""
        indx = list(range(1, len(self.atoms)))
        dists = self.atoms.get_distance(0, indx)
        return np.min(dists)

    def _add_gaussian_at_random_pos(self):
        w = np.random.rand(3)
        cell = self.atoms.get_cell().T
        mu = np.sum(cell.dot(w))
        sigma = 0.0
        for gauss in self.gaussians:
            sigma += gauss.sigma
        sigma /= len(self.gaussians)
        self.add_gaussian(mu, sigma=sigma)

    def find_clusters(self, add_cluster_interval=20, max_iter=10000):
        """Find all clusters from scratch."""
        # Initially we start by a assuming there
        # exists a spherical cluster at the center
        cell = self.atoms.get_cell()
        center = 0.5*np.sum(cell, axis=0)
        self.add_gaussian(center)
        converged = False

        step = 0
        while not converged:
            step += 1
            self.classify()
            self.update_gaussian_parameters()

            if step%add_cluster_interval== 0:
                self._add_gaussian_at_random_pos()

            indx = np.argwhere(self.cluster_id != -1)
            if np.min(self.probability[indx]) > self.threshold:
                return
            if step > max_iter:
                raise CouldNotClassifyClusterError(
                    "Reached maximum number of iterations during "
                    "cluster classification")

    def show_clusters(self):
        from ase.gui.gui import GUI
        from ase.gui.images import Images
        all_clusters = []
        self.tag_by_probability()
        for uid in range(len(self.gaussians)):
            indx = np.argwhere(self.cluster_id == uid)
            cluster = self.atoms[indx]
            cluster.info = {"name": "Cluster ID: {}".format(uid)}
            all_clusters.append(cluster)

        images = Images()
        images.initialize(all_clusters)
        gui = GUI(images)
        gui.show_name = True
        gui.run()

    def tag_by_probability(self):
        """Add the probability of belonging to this cluster
           in the tag."""
        for atom in self.atoms:
            atom.tag = self.probability[atom.index]

    def classify(self):
        """Classify all atoms."""
        self.num_members = [0 for _ in range(len(self.num_members))]
        for atom in self.atoms:
            if atom.symbol not in self.cluster_elements:
                continue
            prob = self.likely_hoods(atom.position)
            uid = np.argmax(prob)
            self.cluster_id[atom.index] = uid
            self.probability[atom.index] = np.max(prob)
            self.num_members[uid] += 1

    def update_gaussian_parameters(self):
        """Update the parameters of the Gaussians."""
        for uid in range(len(self.gaussians)):
            indx = np.argwhere(self.cluster_id==uid)
            pos = self.atoms.get_positions()[indx, :]
            mu = np.mean(pos, axis=0)
            sigma = np.zeros((3, 3))
            mu_muT = np.outer(mu, mu)
            num = 0
            for i in range(pos.shape[0]):
                sigma += np.outer(pos[i, :], pos[i, :]) - mu_muT
                num += 1
            sigma /= num
            
            self.gaussians[uid].mu = mu
            self.gaussians[uid].sigma = sigma
            



