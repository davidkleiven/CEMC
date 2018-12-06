import numpy as np
from cemc.tools import StrainEnergy

class IsotropicStrainEnergy(object):
    def __init__(self, bulk_mod=76.0, shear_mod=26.0):
        self.B = bulk_mod
        self.G= shear_mod
        self.tensor = self._get_isotropic_elastic_tensor(
            bulk_mod, shear_mod)


    def _get_isotropic_elastic_tensor(self, B, G):
        """Return the isotropic elastic tensor (Mandel notation)

        :param float B: Bulk modoulus
        :param flaot G: Shear modulus
        """

        tensor = np.zeros((6, 6))
        tensor[0, 0] = tensor[1, 1] = tensor[2, 2] = \
            B + 4.0*G/3.0
        tensor[0, 1] = tensor[0, 2] = \
        tensor[1, 0] = tensor[1, 2] = \
        tensor[2, 0] = tensor[2, 1] = B - 2.0*G/3.0
        tensor[3, 3] = tensor[4, 4] = tensor[5, 5] = 2*G
        return tensor

    @property
    def poisson(self):
        return 0.5*(3.0*self.B - 2.0*self.G)/(3.0*self.B + self.G)

    def plot(self, princ_misfit=[0.1, 0.1, 0.1], theta=0.0, phi=0.0, show=True):
        from matplotlib import pyplot as plt
        from cemc.tools import rot_matrix_spherical_coordinates
        from cemc.tools import rotate_tensor
        scale_factors = np.logspace(-3, 3, 100)
        
        # Sphere
        aspects = {
            "Sphere": [1.0, 1.0, 1.0],
            "Needle": [10000.0, 1.0, 1.0],
            "Plate":  [10000.0, 10000.0, 1.0]
        }

        fig = plt.figure()
        ax1 = fig.add_subplot(1, 1, 1)
        colors = ["#5D5C61", "#7395AE", "#B1A296"]
        color_count = 0
        for k, v in aspects.items():
            strain_tensor = np.diag(princ_misfit)
            
            rot_matrix = rot_matrix_spherical_coordinates(phi, theta)
            strain_tensor = rotate_tensor(strain_tensor, rot_matrix)

            strain = StrainEnergy(aspect=v, 
                                eigenstrain=strain_tensor, 
                                poisson=self.poisson)
            
            energy = [strain.strain_energy(C_matrix=self.tensor, scale_factor=f)
                    for f in scale_factors]
        
            ax1.plot(scale_factors, energy, label=k, color=colors[color_count%3])
            color_count += 1

        ax1.set_xscale("log")
        ax1.set_yscale("log")
        ax1.spines["right"].set_visible(False)
        ax1.spines["top"].set_visible(False)
        ax1.set_xlabel("Scale factor")
        ax1.set_ylabel("Strain energy")
        ax1.legend(frameon=False)

        if show:
            plt.show()
        