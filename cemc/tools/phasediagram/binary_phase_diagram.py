import dataset
import numpy as np


class BinaryPhaseDiagram(object):
    def __init__(self, db_name=None, fig_prefix=None, table="simulations",
                 phase_id="phase", chem_pot="mu_c1_0", energy="sgc_energy",
                 concentration="singlet_c1_0", temp_col="temperature",
                 tol=1E-6, postproc_table="postproc",
                 recalculate_postproc=False, ht_phases=[], num_elem=2,
                 natoms=1):
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
                res = free_eng.free_energy_isochemical(
                        T=temps, sgc_energy=energies, nelem=self.num_elem)

                c = {self.chem_pot: [conc[i] for i in res["order"]]}
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
                           polyorder=2):
        """Construct a phase separation line between two phases."""
        from cemc.tools.phasediagram import Polynomial
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
                x = mus[p]
            elif temperature is None:
                x = temperatures[p]
            polys[p] = poly.fit(x, grand_pot[p])

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
            fname += "{}_{}_{}K.png".format(phases[0], phases[1], temperature)
            self._create_figure(figname, polys, inter, mus, grand_pot)
        return inter

    def _create_figure(self, figname, polys, x, y, inter=None):
        """Create a figure showing the intersection point in the two phases."""
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
            if np.min(y) < min_y:
                min_y = np.min(y)
            if np.max(y) > max_y:
                max_y = np.max(y)
        x_fit = np.linspace(min_x, max_x, 100)
        for k in polys.keys():
            ax.plot(np.polyval(polys[k], x_fit))
            ax.plot(x[k], y[k], "o")

        if inter is not None:
            k = list(polys.keys())[0]
            ax.plot(inter, np.polyval(polys[k], inter), "*")
        ax.set_ylim([min_y, max_y])
        fig.savefig(figname)

    def _intersection(self, p1, p2, x0):
        """Find the intersection point between two polynomials."""
        from scipy.optimize import fsolve
        res = fsolve(lambda x: np.polyval(p1, x) - np.polyval(p2, x),
                     x0=x0)
        return res[0]

    def phase_boundary(self, phases=[], polyorder=2,
                       variable="chem_pot"):
        """Construct a phase boundary between two phases."""

        allowed_vars = ["chem_pot", "temperature"]
        if variable not in allowed_vars:
            raise ValueError("Variable has to be one of {}"
                             "".format(allowed_vars))

        temperatures = []
        chemical_potentials = []
        if variable == "temperature":
            for mu in self.chem_pots:
                inter = self.phase_intersection(mu=mu, phases=phases,
                                                polyorder=polyorder)
                if inter is not None:
                    temperatures.append(inter)
                    chemical_potentials.append(mu)
        else:
            for t in self.temperatures:
                inter = self.phase_intersection(temperature=t, phases=phases,
                                                polyorder=polyorder)
                if inter is not None:
                    chemical_potentials.append(inter)
                    temperatures.append(t)
        return chemical_potentials, temperatures
