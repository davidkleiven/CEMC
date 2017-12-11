from scipy.stats import linregress
from scipy import interpolate
import numpy as np
from scipy.misc import derivative
from matplotlib import pyplot as plt

class SGCToCanonicalConverter(object):
    def __init__( self, wl_analyzers, natoms ):
        if ( len(wl_analyzers) < 3 ):
            raise ValueError( "DOS for at least 3 chemical potentials has to be provided" )
        self.wl_analyzers = wl_analyzers
        self.n_atoms = natoms
        self.composition = None
        self.chemical_potentials = None
        self.free_energies = None
        self.all_chem_pots = None
        self.normalize_dos()

    def normalize_dos( self ):
        """
        Normalize the DOS in each wl_simulation
        """
        factor = 1.0/self.wl_analyzers[0].dos[0]
        print (self.wl_analyzers[0].chem_pot)
        for wl in self.wl_analyzers:
            #factor = 1.0
            wl.dos *= factor


    def hyper_surface_in_chemical_potential_space( self, T, points ):
        """
        Generates a hyper surface representing the SGC potential
        """
        if ( self.free_energies is None or self.all_chem_pots is None ):
            all_chem_pots = []
            # Find all symbols in the simulation
            symbs = self.wl_analyzers[0].chem_pots.keys()
            indx = {key:i for i in range(symbs)}

            # Extract chemical potentials
            sgc_pots = []
            for wl in self.wl_analyzers:
                chem_pots = np.zeros(len(len(symbs)))
                for symb in symbs:
                    chem_pots[indx[symb]] = wl.chem_pots[symb]
                all_chem_pots.append(chem_pots)
                sgc_pots.append(wl.free_energy(T))
            self.free_energies = sgc_pots

            # Build surface
            self.all_chem_pots = np.array(all_chem_pots)

        sgc_pot_surface = interpolate.griddata( self.all_chem_pots, self.free_energies, points, method="cubic" )
        sgc_pot_surface = interpolate.LinearNDInterpolator( self.all_chem_pots, self.free_energies )
        return sgc_pot_surface

    def get_composition_one_element( self, T, element, spline_order=3, n_chem_pots=50 ):
        """
        Computes the compositions. DOES NOT WORK AT THE MOMENT, HAS TO USE hyper_surface_in_chemical_potential_space
        """
        chem_pots = []
        thermo_potentials = []
        for wl in self.wl_analyzers:
            chem_pots.append(wl.chem_pot[element])
            thermo_potentials.append( wl.free_energy(T) )

        # Sort the chemical potentials
        sort_arg = np.argsort(chem_pots)
        sorted_chem_pots = [chem_pots[indx] for indx in sort_arg]
        sorted_thermo_pots = [thermo_potentials[indx] for indx in sort_arg]
        if ( np.allclose(sorted_chem_pots, 0.0) ):
            # This is the reference element
            x = np.zeros(n_chem_pots)
            phi = np.zeros(n_chem_pots)
            new_chem_pots = np.zeros(n_chem_pots)
            return new_chem_pots, x, phi, sorted_chem_pots, sorted_thermo_pots

        interpolator = interpolate.interp1d(sorted_chem_pots,sorted_thermo_pots,kind="linear", bounds_error=False, fill_value="extrapolate")
        #interpolator = interpolate.PchipInterpolator(sorted_chem_pots,sorted_thermo_pots,extrapolate=True)
        new_chem_pots = np.linspace(np.min(sorted_chem_pots), np.max(sorted_chem_pots), n_chem_pots)
        d_mu = new_chem_pots[1]-new_chem_pots[0]
        x = -derivative(interpolator, new_chem_pots, d_mu )/self.n_atoms
        #x[x<0.0] = 0.0
        #x[x>1.0] = 1.0
        phi = interpolator(new_chem_pots)
        return new_chem_pots, x, phi, sorted_chem_pots, sorted_thermo_pots

    def get_compositions( self, T, spline_order=3, n_chem_pots=50 ):
        """
        Get the composition of all elements.
        NOTE: One of the elements will have a composition of zero since its
        chemical potential was used as a reference.
        The proper composition of this element is 1 minus the composition
        of all the others
        """
        elms = self.wl_analyzers[0].chem_pot.keys()
        comp = {}
        sgc_pot = {}
        chem_pot = {}
        chem_pot_raw = {}
        sgc_pot_raw = {}
        for symbol in elms:
            chem_pot[symbol], comp[symbol], sgc_pot[symbol], chem_pot_raw[symbol], sgc_pot_raw[symbol] = self.get_composition_one_element( T, symbol, spline_order=spline_order )
        self.composition = comp
        self.free_energies = sgc_pot
        self.chemical_potentials = chem_pot
        return chem_pot, comp, sgc_pot, chem_pot_raw, sgc_pot_raw

    def free_energy( self, T ):
        """
        Return the Helmholtz free energy
        """

        self.get_compositions(T)

        helmholtz_free_energy = []
        interpolators = {elm:interpolate.interp1d(self.composition[elm], self.chemical_potentials[elm]) for elm in self.chemical_potentials.keys()}

        # TODO: This is really a hyper surface so it should be a multidimensional surface
        free_eng_interp = interpolate.interp1d( self.composition["Cu"], self.free_energies["Cu"] )

        for comp in self.composition["Cu"]:
            helmholtz_free_energy.append( free_eng_interp(comp)/self.n_atoms )
            helmholtz_free_energy[-1] += interpolators["Cu"]( comp )*comp
        return self.composition["Cu"], helmholtz_free_energy

    def binary_diagram( self, T, elm1, elm2, use_inverse_for_elm2=True ):
        """
        Plots a binary phase diagram
        """
        helm_holtz_energies = {}
        comps = []
        free_energies = []
        all_temps = []
        for temp in T:
            self.get_compositions(temp)
            comps += list(self.composition[elm1])
            c,h = self.free_energy(temp)
            free_energies += list(h)
            all_temps += [temp for _ in range(len(self.composition[elm1]))]

        comps = np.array( comps )
        all_temps = np.array(all_temps)
        free_energies = np.array( free_energies )
        print (len(comps),len(all_temps),len(free_energies))
        comp = np.linspace(0.0,1.0,100)
        temperatures = np.linspace(np.min(all_temps),np.max(all_temps),100)
        X,Y = np.meshgrid(comp,temperatures)
        points = np.vstack((comps,all_temps)).T
        free_eng = interpolate.griddata( points, free_energies, (X,Y), method="cubic" )

        # Plot iso-curves for the Free energy
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        im = ax.contour( X, Y, free_eng, 20 )
        fig.colorbar(im)
        return fig
