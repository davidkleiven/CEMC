import dataset
import numpy as np
from cemc.tools.phasediagram import SingleCurvaturePolynomial


class BinaryPhaseDiagram(object):
    """Class for constructing binary phase diagrams from
        grand canonical calculations.

    :param str db_name: Database containing the MC calculations
    :param str fig_prefix: If given, images of of all polynomial
        fits and intersection points will be stored with this
        prefix
    :param str table: Name of the table in the database with MC
        result
    :param str phase_id: Name of the column holding informaiton about
        which phase the simulation belongs to.
    :param str chem_pot: Name of the column with the chemical potentials
    :param str energy: Name of the column with the energy
    :param str concentration: Name of the column with the variable storing
        information about the concentration
    :param str temp_col: Name of the column with temperature
    :param float tol: Tolerance used when making queries on floating point
        numbers in the database
    :param str postproc_table: Name of the table where post processing
        information will be stored-
    :param bool recalculate_postproc: If True, the postprocessing table
        will be recalculated even if it already exist.
    :param list ht_phases: List with the name of phases that should
        be integrated from the high temperature expansion. The Grand
        Potential for all other phases will be obtained by integration
        from a low temperature expansion.
    :param int num_elem: Number of elements in the system
    :param int natoms: Number of atoms in the simulations. This number
        is used to normalize the energy in the database. So if the
        energy in the database is already normalized, this number
        should be set to 1.
    """
    def __init__(self, db_name=None, fig_prefix=None, table="simulations",
                 phase_id="phase", chem_pot="mu_c1_0", energy="sgc_energy",
                 concentration="singlet_c1_0", temp_col="temperature",
                 tol=1E-6, postproc_table="postproc",
                 recalculate_postproc=False, ht_phases=[], num_elem=2,
                 natoms=1, isochem_ref=None, num_per_fu=1):
        self.db_name = db_name
        self.fig_prefix = fig_prefix
        self.table = table
        self.phase_id = phase_id
        self.chem_pot = chem_pot
        self.energy = energy
        self.concentration = concentration
        self.temp_col = temp_col
        self.tol = tol
        self.postproc_table = postproc_table
        self.recalculate_postproc = recalculate_postproc
        self.ht_phases = ht_phases
        self.num_elem = num_elem
        self.natoms = natoms
        self.isochem_ref = isochem_ref
        self.num_per_fu = num_per_fu

        self.temperatures = self._get_temperatures()
        self.chem_pots = self._get_chemical_potentials()
        self.all_phases = self._get_all_phases()
        self._grand_potential()

    def _get_temperatures(self):
        """Read all temperatures from the database."""
        db = dataset.connect(self.db_name)
        tbl = db[self.table]
        temps = []
        for row in tbl.find():
            temps.append(row[self.temp_col])
        return list(set(temps))

    def _get_chemical_potentials(self):
        """Read all chemical potentials from the database."""
        db = dataset.connect(self.db_name)
        tbl = db[self.table]
        mu = []
        for row in tbl.find():
            mu.append(row[self.chem_pot])
        return list(set(mu))

    def _get_all_phases(self):
        """Read all phases in the database."""
        db = dataset.connect(self.db_name)
        tbl = db[self.table]
        phases = set()
        for row in tbl.find():
            phases = phases.union([row[self.phase_id]])
        return list(phases)

    def _grand_potential(self):
        """Calculate the grand potential and put in a separate table."""
        from cemc.tools import FreeEnergy
        db = dataset.connect(self.db_name)
        if self.postproc_table in db.tables and not self.recalculate_postproc:
            return

        tbl_in = db[self.table]
        tbl_pp = db[self.postproc_table]
        for ph in self.all_phases:
            for mu in self.chem_pots:
                ids = []
                energies = []
                temps = []
                conc = []
                query = {self.chem_pot: {"between":
                                         [mu-self.tol, mu+self.tol]},
                         self.phase_id: ph}

                for row in tbl_in.find(**query):
                    ids.append(row["id"])
                    energies.append(row[self.energy]/self.natoms)
                    temps.append(row[self.temp_col])
                    conc.append(row[self.concentration])

                if len(ids) == 0:
                    continue

                if ph in self.ht_phases:
                    limit = "hte"
                else:
                    limit = "lte"
                free_eng = FreeEnergy(limit=limit)
                ref = self.isochem_ref.get(ph, None)
                if ref is not None:
                    ref = self.isochem_ref[ph](mu)

                res = free_eng.free_energy_isochemical(
                        T=temps, sgc_energy=energies, nelem=self.num_elem,
                        beta_phi_ref=ref)

                c = {self.chem_pot: [conc[i]*self.num_per_fu for i in res["order"]]}
                ch = {self.chem_pot: mu}
                phi = res["free_energy"]

                gibbs = free_eng.helmholtz_free_energy(phi, c, ch)
                # Write the results to the database
                ids_srt = [ids[x] for x in res["order"]]
                for i in range(len(res["temperature"])):
                    entry = {"systemID": ids_srt[i],
                             "grand_potential": res["free_energy"][i],
                             "free_energy": gibbs[i]}
                    tbl_pp.upsert(entry, ["systemID"])

    def phase_intersection(self, temperature=None, mu=None, phases=[],
                           polyorder=2, bounds={}):
        """Construct a phase separation line between two phases.

        :param float temperature: If given, a phase boundary at
            this fixed temperature is sought.
        :param float mu: If given, a phase boundary at this fixed
            chemical potential is sought.
        :param list phases: List of length 2 describing which
            phases a phase boundary should be found.
        :param int polyorder: Order of the polynomial used to
            fit the data.
        """
        from cemc.tools.phasediagram import Polynomial
        if len(phases) != 2:
            raise ValueError("Two phases has to be given!")
        db = dataset.connect(self.db_name)
        tbl = db[self.table]
        tbl_pp = db[self.postproc_table]
        ids = {p: [] for p in phases}
        mus = {p: [] for p in phases}
        temperatures = {p: [] for p in phases}
        grand_pot = {p: [] for p in phases}

        if mu is None and temperature is None:
            raise ValueError("Temperature or chemical potential has to "
                             "be specified")

        # Retrieve the IDs
        for ph in phases:
            if mu is None:
                query = {self.temp_col: {'between': [temperature-self.tol,
                                                     temperature+self.tol]},
                         self.phase_id: ph}
            elif temperature is None:
                query = {self.chem_pot: {'between': [mu-self.tol,
                                                     mu+self.tol]},
                         self.phase_id: ph}

            for row in tbl.find(**query):
                ids[ph].append(row["id"])
                mus[ph].append(row[self.chem_pot])
                temperatures[ph].append(row[self.temp_col])

        for ph in phases:
            if len(ids[ph]) <= 2:
                return None
            elif len(ids[ph]) == 3:
                polyorder = 1

        # Extract the grand potential
        for ph in phases:
            for systID in ids[ph]:
                row = tbl_pp.find_one(systemID=systID)
                grand_pot[ph].append(row["grand_potential"])

        # Look for intersection point
        polys = {p: None for p in phases}
        for p in phases:
            if len(grand_pot[p]) < polyorder+2:
                new_polyorder = len(grand_pot[p])-2
                if new_polyorder < 0:
                    new_polyorder = 0
                print("Warning! Too few points. Reducing polynomial "
                      "order from {} to {}".format(polyorder, new_polyorder))
                polyorder = new_polyorder

            poly = Polynomial(order=polyorder)

            if mu is None:
                x = np.array(mus[p])
            elif temperature is None:
                x = np.array(temperatures[p])

            G = np.array(grand_pot[p])

            limits = bounds.get(p, None)
            if limits is not None:
                G = G[x >= limits[0]]
                x = x[x >= limits[0]]
                G = G[x < limits[1]]
                x = x[x < limits[1]]

            if len(x) < polyorder+2:
                return None

            polys[p] = poly.fit(x, G)

        # Find intersection point
        x0 = 0.0
        if mu is None:
            for _, v in mus.items():
                x0 += np.mean(v)
        elif temperature is None:
            for _, v in temperatures.items():
                x0 += np.mean(v)

        x0 /= 2
        inter = self._intersection(polys[phases[0]], polys[phases[1]], x0)
        if self.fig_prefix is not None:
            fname = self.fig_prefix
            if mu is None:
                fname += "{}_{}_{}K.png".format(phases[0], phases[1], int(temperature))
                self._create_figure(fname, polys, mus, grand_pot, inter=inter)
            elif temperature is None:
                fname += "{}_{}_{}K.png".format(phases[0], phases[1], mu)
                self._create_figure(fname, polys, temperatures, grand_pot, inter=inter)
        return inter

    def _create_figure(self, figname, polys, x, y, inter=None):
        """Create a figure showing the intersection point in the two phases.

        :param str figname: Filename for the generated figure
        :param dict polys: Dictionary of the same form as x (see below),
            but holding the coefficients for the fitted polynomials
        :param dict x: Dictionary with the form
            {
                phase1: [x11, x12, x13, x14, ...],
                phase2: [x21, x22, x23, x24, ...]
            }
        :param dict y: Dictionary with the same form as
            x, but holding the y-values instead
        :param float inter: If given, this point will be plotted
            as a star to highlight the intersection point.
        """
        from matplotlib import pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        min_x = 1E100
        max_x = -1E100
        for k, v in x.items():
            if np.min(v) < min_x:
                min_x = np.min(v)
            if np.max(v) > max_x:
                max_x = np.max(v)

        min_y = 1E100
        max_y = -1E100
        for k, v in y.items():
            if np.min(v) < min_y:
                min_y = np.min(v)
            if np.max(v) > max_y:
                max_y = np.max(v)
        x_fit = np.linspace(min_x, max_x, 100)
        for k in polys.keys():
            ax.plot(x_fit, np.polyval(polys[k], x_fit))
            ax.plot(x[k], y[k], "o")

        if inter is not None:
            k = list(polys.keys())[0]
            ax.plot(inter, np.polyval(polys[k], inter), "*")
        diff = max_y - min_y
        ax.set_ylim([min_y-0.05*diff, max_y + 0.05*diff])
        fig.savefig(figname)
        plt.close()

    def _intersection(self, p1, p2, x0):
        """Find the intersection point between two polynomials.

        :param list p1: Coefficients for the first polynomial
        :param list p2: Coefficients for the second polynomial
        :param float x0: Initial guess for the intersection point
        """
        from scipy.optimize import fsolve
        res = fsolve(lambda x: np.polyval(p1, x) - np.polyval(p2, x),
                     x0=x0)
        return res[0]

    def phase_boundary(self, phases=[], polyorder=2,
                       variable="chem_pot", bounds={}):
        """Construct a phase boundary between two phases.

        :param list phases: List with the name of the phases where
            the phase boundary should be found.
        :param int polyorder: Order the polynomials used in the fit
        :param str variable: Which is the free variable. Has to
            be one of [temperature, chem_pot]
        """

        allowed_vars = ["chem_pot", "temperature"]
        if variable not in allowed_vars:
            raise ValueError("Variable has to be one of {}"
                             "".format(allowed_vars))

        temperatures = []
        chemical_potentials = []
        if variable == "temperature":
            for mu in self.chem_pots:
                inter = self.phase_intersection(mu=mu, phases=phases,
                                                polyorder=polyorder,
                                                bounds=bounds)
                if inter is not None:
                    temperatures.append(inter)
                    chemical_potentials.append(mu)
        else:
            for t in self.temperatures:
                inter = self.phase_intersection(temperature=t, phases=phases,
                                                polyorder=polyorder,
                                                bounds=bounds)
                if inter is not None:
                    chemical_potentials.append(inter)
                    temperatures.append(t)
        return chemical_potentials, temperatures

    def composition(self, phase, temperature=None, mu=None, polyorder=2,
                    bounds=None, interp=None):
        """Get the composition in a given phase

        :param float temperature: Temperature
        :param list mu: List of chemical potentials where
            the phase is sought.
        :param str phase: Phase
        :param int polyorder: Order of the fitted polynomial
        """

        if temperature is None and mu is None:
            raise ValueError("Temperature or mu has to be given!")

        db = dataset.connect(self.db_name)
        tbl = db[self.table]

        if mu is None:
            query = {
                self.temp_col: {"between": [temperature-self.tol,
                                            temperature+self.tol]},
                self.phase_id: phase
            }
        else:
            query = {
                self.chem_pot: {"between": [mu-self.tol,
                                            mu+self.tol]},
                self.phase_id: phase
            }

        mu_db = []
        temp_db = []
        conc = []
        for row in tbl.find(**query):
            mu_db.append(row[self.chem_pot])
            temp_db.append(row[self.temp_col])
            conc.append(row[self.concentration])

        if mu is None:
            x = mu_db
        else:
            x = temp_db
        
        x = np.array(x)
        conc = np.array(conc)
        if bounds is not None:
            conc = conc[x >= bounds[0]]
            x = x[x >= bounds[0]]
            conc = conc[x < bounds[1]]
            x = x[x < bounds[1]]
            
        p = np.polyfit(x, conc, polyorder)

        if self.fig_prefix is not None:
            from matplotlib import pyplot as plt
            figname = self.fig_prefix
            if mu is None:
                figname += "comp_{}K_{}.png".format(temperature, phase)
            elif temperature is None:
                figname += "comp_{}eV_{}.png".format(mu, phase)
            combined = sorted(zip(x, conc))
            x, conc = zip(*combined)
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            x_fit = np.linspace(np.min(x), np.max(x), 50)
            ax.plot(x, conc, marker="o", mfc="none")
            ax.plot(x_fit, np.polyval(p, x_fit))
            ax.set_xlabel("Chemical potential")
            ax.set_ylabel("Composition")
            fig.savefig(figname)
            plt.close()
        return p

    def _guess_curvature(self, x, y):
        """Guess if the curve is convex or concave."""
        coeff = np.polyfit(x, y, 2)
        if coeff[0] > 0.0:
            return "convex"
        return "concave"

    def _guess_slope(self, x, y):
        """Guess if the curve is increasing or decreasing."""
        coeff = np.polyfit(x, y, 1)
        if coeff[0] > 0.0:
            return "increasing"
        return "decreasing"
