import numpy as np
from itertools import combinations_with_replacement
import time
from random import choice

class CouldNotClassifyClusterError(Exception):
    pass

class GaussianClusterTracker(object):
    def __init__(self, atoms=None, threshold=0.001, 
                 cluster_elements=[], num_clusters=1, init_centroids=[]):
        self.atoms = atoms
        self._nn_dist= self._nn_distance()
        self.threshold = threshold
        self.cluster_id = -np.ones(len(self.atoms), dtype=np.int8)
        self.probability = np.zeros(len(self.atoms))
        self.gaussians = []
        self.num_members = []
        self.prob_belong = np.zeros((len(atoms), num_clusters))
        self.cluster_elements = cluster_elements

        if init_centroids:
            if len(init_centroids) != num_clusters:
                raise ValueError("The length of the centroids, "
                                 "must match the number of clusters.")
            for centroid in init_centroids:
                self.add_gaussian(centroid)
        else:
            for _ in range(num_clusters):
                indx = choice(self.solute_indices)
                self.add_gaussian(self.atoms[indx].position)

        self.frac_per_cluster = np.zeros(num_clusters) + 1.0/num_clusters
        self.output_every = 20
        self._check_input()
        self.penalty = self._nn_distance()**2

    def add_gaussian(self, mu, sigma=None):
        from cemc.tools import MultivariateGaussian
        if sigma is None:
            sigma = np.eye(3)*self._nn_distance()**2

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

    @property
    def solute_indices(self):
        return [atom.index for atom in self.atoms if atom.symbol in self.cluster_elements]

    @property
    def num_solute_atoms(self):
        return sum(1 for atom in self.atoms if atom.symbol in self.cluster_elements)

    def get_cluster(self, cluster_id):
        """Return an atomic cluster."""
        return np.nonzero(self.cluster_id==cluster_id)[0]

    def likelihoods(self, x):
        """Compute all the likely hoods for all distribution."""
        prob = []
        for gauss in self.gaussians:
            prob.append(gauss(x))
        return prob

    def recenter_gaussians(self):
        """If a gaussian has no member. Recenter it to random location."""
        for i, gaussian in enumerate(self.gaussians):
            if self.num_members[i] == 0:
                indx = choice(self.solute_indices)
                gaussian.mu = self.atoms[indx].position

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
        # Currently only works with one cluster
        assert self.num_clusters == 1

        for change in system_changes:
            x = self.atoms[change[0]].position
            if change[1] in self.cluster_elements and change[2] not in self.cluster_elements:
                # One atom is removed
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
                    mu2_mu2T = np.outer(mu2, mu2)
                    sigma2 = (N*sigma1 + N*mu1_mu1T - np.outer(x, x))/(N-1) - mu2_mu2T - np.eye(3)*self.penalty/(N-1)

                self.num_members[old_uid] = N-1
                self.gaussians[old_uid].mu = mu2
                self.gaussians[old_uid].sigma = sigma2
                self.cluster_id[old_uid] = -1
            elif change[1] not in self.cluster_elements and change[2] in self.cluster_elements:
                # One atom is added to the cluster
                prob = self.likelihoods(self.atoms[change[0]].position)
                uid = np.argmax(prob)
                self.cluster_id[change[0]] = uid
                N = self.num_members[uid]
                mu = self.gaussians[uid].mu
                sigma1 = self.gaussians[uid].sigma
                mu2 = (N*mu + x)/(N+1)
                mu1_mu1T = np.outer(mu, mu)
                mu2_mu2T = np.outer(mu2, mu2)
                sigma2 = (N*sigma1 + N*mu1_mu1T + np.outer(x, x))/(N+1) - mu2_mu2T + np.eye(3)*self.penalty/(N+1)
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
        sigma = np.eye(3)*self.penalty
        self.add_gaussian(mu, sigma=sigma)

    def find_clusters(self, max_iter=10000):
        """Find all clusters from scratch."""
        # Initially we start by a assuming there
        # exists a spherical cluster at the center
        converged = False
        step = 0
        now = time.time()
        prev_cost = 100000000000000.0
        step = 0
        print("here")
        while not converged:
            print(self.gaussians[0].mu)
            if step > 1:
                self.recenter_gaussians()

            step += 1
            if time.time() - now > self.output_every:
                print("Cost: {:.2e}. Iteration: {}".format(prev_cost, step))
                now = time.time()

            # Expectation step
            self._calc_belong_prob()

            # Maximation step
            m_c = np.sum(self.prob_belong, axis=0)
            self.frac_per_cluster = m_c/self.num_solute_atoms
            self.set_mu_sigma()
            cost = self.log_likelihood()
            self.classify()
            if abs(cost - prev_cost) < self.threshold:
                print("Final log-likelihood: {:.2e}".format(cost))
                return
            prev_cost = cost

            if step >= max_iter:
                return
            
    def set_mu_sigma(self):
        """Calculate new values for sigma and mu."""
        for i, gaussian in enumerate(self.gaussians):
            r = self.prob_belong[:, i]
            m_c = np.sum(r)
            pos = self.atoms.get_positions()
            mu = pos.T.dot(r)/m_c

            pos -= mu
            sigma = pos.T.dot(np.diag(r)).dot(pos)/m_c

            sigma += np.eye(3)*self.penalty
            gaussian.mu = mu
            gaussian.sigma = sigma

    def log_likelihood(self):
        log_likelihood = 0.0
        count = 0
        for atom in self.atoms:
            if atom.symbol not in self.cluster_elements:
                continue

            likeli = np.array(self.likelihoods(atom.position))
            log_likelihood += np.log(np.sum(self.frac_per_cluster*likeli))
            count += 1
        return log_likelihood/count

    def _calc_belong_prob(self):
        self.prob_belong[:, :] = 0.0
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

    def show_gaussians(self, scale=1.0, r=1.0, show=True):
        """Show clusters as ellipsoids.
        
        :param float scale: Number of standard deviations to show
        :param float r: Thickness of the tubes used to represent
            the unit cell
        :param bool show: If True mlab.show() is executed
        """
        from mayavi.api import Engine
        from mayavi import mlab
        engine = Engine()
        engine.start()
        scene = engine.new_scene()
        scene.scene.disable_render = True # for speed
        surfs = []
        self._draw_cell(r=r)
        for gauss in self.gaussians:
            surf = self._show_one_gaussian(gauss, engine, scale=scale)
            surfs.append(surf)

        scene.scene.disable_render = False
        for i, surf in enumerate(surfs):
            vtk_srcs = mlab.pipeline.get_vtk_src(surf)
            vtk_src = vtk_srcs[0]
            npoints = len(vtk_src.point_data.scalars)
            vtk_src.point_data.scalars = np.tile(i, npoints)

        if show:
            mlab.show()

    def _draw_cell(self, color=(0, 0, 0), r=1.0):
        from mayavi import mlab
        cell = self.atoms.get_cell()
        for i in range(3):
            x = [0, cell[i, 0]]
            y = [0, cell[i, 1]]
            z = [0, cell[i, 2]]
            mlab.plot3d(x, y, z, color=color, tube_radius=r)

            x = [cell[i, 0], cell[i, 0] + cell[(i+1)%3, 0]]
            y = [cell[i, 1], cell[i, 1] + cell[(i+1)%3, 1]]
            z = [cell[i, 2], cell[i, 2] + cell[(i+1)%3, 2]]
            mlab.plot3d(x, y, z, color=color, tube_radius=r)

            x = [cell[i, 0], cell[i, 0] + cell[(i+2)%3, 0]]
            y = [cell[i, 1], cell[i, 1] + cell[(i+2)%3, 1]]
            z = [cell[i, 2], cell[i, 2] + cell[(i+2)%3, 2]]
            mlab.plot3d(x, y, z, color=color, tube_radius=r)

            x = [cell[i, 0] + cell[(i+1)%3, 0], cell[i, 0] + cell[(i+1)%3, 0] + cell[(i+2)%3, 0]]
            y = [cell[i, 1] + cell[(i+1)%3, 1], cell[i, 1] + cell[(i+1)%3, 1] + cell[(i+2)%3, 1]]
            z = [cell[i, 2] + cell[(i+1)%3, 2], cell[i, 2] + cell[(i+1)%3, 2] + cell[(i+2)%3, 2]]
            mlab.plot3d(x, y, z, color=color, tube_radius=r)

    def _show_one_gaussian(self, gauss, engine, scale=1.0):
        """Plot one of the gaussians."""
        from mayavi.sources.api import ParametricSurface
        from mayavi.modules.api import Surface
        
        source = ParametricSurface()
        source.function = 'ellipsoid'
        engine.add_source(source)

        eigval, eigvec = np.linalg.eig(gauss.sigma)

        angles = rotationMatrixToEulerAngles(eigvec.T)*180.0/np.pi
        surface = Surface()
        source.add_module(surface)
        actor = surface.actor

        actor.property.opacity = 0.5
        actor.property.color = tuple(np.random.rand(3))
        actor.mapper.scalar_visibility = False
        actor.property.backface_culling = True
        actor.actor.orientation = np.array([0.0, 0.0, 0.0])
        actor.actor.origin = np.array([0.0, 0.0, 0.0])
        actor.actor.position = np.array(gauss.mu)
        actor.actor.scale = np.array(scale*np.sqrt(eigval))
        actor.actor.rotate_x(angles[0])
        actor.actor.rotate_y(angles[1])
        actor.actor.rotate_z(angles[2])
        return surface

    def tag_by_probability(self):
        """Add the probability of belonging to this cluster
           in the tag."""

        # Put the weights in the momenta entry
        self.atoms.arrays["initial_charges"] = self.probability

    def classify(self):
        """Classify all atoms."""
        self.num_members = [0 for _ in range(len(self.num_members))]
        self.cluster_id[:] = -1
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
            
def rotationMatrixToEulerAngles(R) :
    sy = np.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
     
    singular = sy < 1e-6
 
    if  not singular :
        x = np.arctan2(R[2,1] , R[2,2])
        y = np.arctan2(-R[2,0], sy)
        z = np.arctan2(R[1,0], R[0,0])
    else :
        x = np.arctan2(-R[1,2], R[1,1])
        y = np.arctan2(-R[2,0], sy)
        z = 0
 
    return np.array([x, y, z])


