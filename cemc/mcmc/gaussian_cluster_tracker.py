import numpy as np
from itertools import combinations_with_replacement
import time

class CouldNotClassifyClusterError(Exception):
    pass

class GaussianClusterTracker(object):
    def __init__(self, atoms=None, threshold=0.001, 
                 cluster_elements=[], num_clusters=1):
        self.atoms = atoms
        self._nn_dist= self._nn_distance()
        self.threshold = threshold
        self.cluster_id = -np.ones(len(self.atoms), dtype=np.int8)
        self.probability = np.zeros(len(self.atoms))
        self.gaussians = []
        self.num_members = []
        self.prob_belong = np.zeros((len(atoms), num_clusters))

        cell = self.atoms.get_cell().T
        for _ in range(num_clusters):
            w = np.random.rand(3)
            mu = cell.dot(w)
            self.add_gaussian(mu)

        self.cluster_elements = cluster_elements
        self.frac_per_cluster = np.zeros(num_clusters) + 1.0/num_clusters
        self.num_clusters = num_clusters
        self._check_input()

    def add_gaussian(self, mu, sigma=None):
        from cemc.tools import MultivariateGaussian
        if sigma is None:
            length = 2.0*np.max(self.atoms.get_cell_lengths_and_angles()[:3])
            sigma = np.eye(3)*length**2

        self.gaussians.append(MultivariateGaussian(mu=mu, sigma=sigma))
        self.num_members.append(0)

    def _check_input(self):
        """Perform some checks on the users input."""
        if self.num_cluster_elements <= 1:
            raise ValueError("There is only one cluster element present!")

    @property
    def num_cluster_elements(self):
        return sum(1 for atom in self.atoms 
                   if atom.symbol in self.cluster_elements)

    @property
    def num_clusters(self):
        return len(self.gaussians)

    def likelihoods(self, x):
        """Compute all the likely hoods for all distribution."""
        prob = []
        for gauss in self.gaussians:
            prob.append(gauss(x))
        return prob

    def move_create_new_cluster(self, system_changes):
        """Check if the proposed move create a new cluster."""
        for change in system_changes:
            if change[2] in self.cluster_elements:
                prob = self.likelihoods(self.atoms[change[0]].positions)
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
                prob = self.likelihoods(self.atoms[change[0]].position)
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
        dists = self.atoms.get_distances(0, indx)
        return np.min(dists)

    def add_new_cluster(self):
        # Find one atom that is not well classified
        prob = np.argmin(self.probability[self.cluster_id != -1])
        indx = np.argmin(np.abs(prob - self.probability))
        mu = self.atoms[indx].position
        sigma = np.eye(3)*self._nn_dist**2
        self.add_gaussian(mu, sigma=sigma)

    def find_clusters(self, max_iter=10000):
        """Find all clusters from scratch."""
        # Initially we start by a assuming there
        # exists a spherical cluster at the center
        cell = self.atoms.get_cell()
        center = 0.5*np.sum(cell, axis=0)
        #self.add_gaussian(center)
        converged = False

        step = 0
        now = time.time()
        prev_cost = 100000000000000.0
        step = 0
        while not converged:
            step += 1
            # Expectation step
            self._calc_belong_prob()

            # Maximation step
            m_c = np.sum(self.prob_belong, axis=0)
            self.frac_per_cluster = m_c/self.prob_belong.shape[0]
            self.set_mu_sigma()
            cost = self.log_likelihood()

            if abs(cost - prev_cost) < self.threshold:
                return

            if step >= max_iter:
                return
            
    def set_mu_sigma(self):
        """Calculate new values for sigma and mu."""
        for i, gaussian in enumerate(self.gaussians):
            m_c = np.sum(self.prob_belong[:, i])
            mu = np.sum(self.prob_belong[:, i]*self.atoms.get_positions())
            mu /= m_c

            pos = self.atoms.get_positions() - mu
            r = self.atoms.get_positions()
            sigma = np.sum(r*pos.T.dot(pos))/m_c
            gaussian.mu = mu
            gaussian.sigma = sigma

    def log_likelihood(self):
        log_likelihood = 0.0
        count = 0
        for atom in self.atoms:
            if atom not in self.cluster_elements:
                continue

            likeli = np.array(self.likelihoods(atom.position))
            log_likelihood += np.log(np.sum(self.frac_per_cluster*likeli))
            count += 1
        return log_likelihood/count



    def _calc_belong_prob(self):
        for atom in self.atoms:
            if atom.symbol not in self.cluster_elements:
                continue
            likeli = np.array(self.likelihoods(atom.position))
            r = self.frac_per_cluster*likeli/(np.sum(self.frac_per_cluster*likeli))
            self.prob_belong[atom.index, :] = r


    def show_clusters(self):
        from ase.gui.gui import GUI
        from ase.gui.images import Images
        all_clusters = []
        self.tag_by_probability()
        for uid in range(len(self.gaussians)):
            cluster = self.atoms[self.cluster_id == uid]
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

        # Put the weights in the momenta entry
        self.atoms.arrays["initial_charges"] = self.probability

    def classify(self):
        """Classify all atoms."""
        self.num_members = [0 for _ in range(len(self.num_members))]
        for atom in self.atoms:
            if atom.symbol not in self.cluster_elements:
                continue
            #prob = self.likelihoods(atom.position)
            uid = np.argmax(self.prob_belong[atom.index, :])
            self.cluster_id[atom.index] = uid
            self.probability[atom.index] = np.max(np.max(self.prob_belong[atom.index, :]))
            self.num_members[uid] += 1

    def update_gaussian_parameters(self):
        """Update the parameters of the Gaussians."""
        for uid in range(len(self.gaussians)):
            pos = self.atoms.get_positions()[self.cluster_id==uid, :]
            if pos.shape[0] >= 2:
                mu = np.mean(pos, axis=0)
                sigma = np.zeros((3, 3))
                mu_muT = np.outer(mu, mu)
                for i in range(pos.shape[0]):
                    sigma += np.outer(pos[i, :], pos[i, :]) - mu_muT
                sigma /= pos.shape[0]
                self.gaussians[uid].mu = mu
                self.gaussians[uid].sigma = sigma
            



