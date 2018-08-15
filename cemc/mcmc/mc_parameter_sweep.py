import h5py as h5
from cemc.mcmc.sgc_montecarlo import SGCMonteCarlo
import numpy as np


class MCParameterSweep(object):
    known_parameters = {
        "MonteCarlo": ["temperature", "composition"],
        "SGCMonteCarlo": ["temperature", "chemical_potential"]
    }

    def __init__(self, parameters, mc_obj, nsteps=100000, data_getter=None,
                 outfile="default_output.h5", equil_params=None):
        self.parameters = parameters
        self.mc_obj = mc_obj
        self.nsteps = nsteps
        self.data_getter = None
        self.check_initialization()
        self.outfile = outfile
        self.equil_params = equil_params

    def check_initialization(self):
        """
        Check that the parameters are valid
        """
        if self.mc_obj.name not in self.known_parameters.keys():
            msg = "Monte Carlo instance was not recognized. Known MonteCarlo "
            msg += "object: {}".format(self.known_parameters.keys())
            raise ValueError(msg)

        required_params = self.known_parameters[self.mc_obj.name]

        for i in range(len(self.parameters)):
            for req_param in required_params:
                if req_param not in self.parameters[i].keys():
                    msg = "Required parameter {} ".format(req_param)
                    msg += "not given for entry {}.".format(i)
                    raise ValueError(msg)

            # Check the format of the different parameters
            if "chemical_potential" in self.parameters[i].keys():
                if not isinstance(self.parameters[i]["chemical_potential"],
                                  dict):
                    msg = "Chemical potential has to be given as a dictionary"
                    raise ValueError(msg)

            if "temperature" in self.parameters[i].keys():
                try:
                    T = float(self.parameters[i]["temperature"])
                except:
                    raise ValueError("Temperature has to be given as a float")

    def run(self):
        n_params = len(self.parameters)
        name = self.mc_obj.name
        all_data = []
        for i in range(n_params):
            self.mc_obj.reset()
            if name == "SGCMonteCarlo":
                self.mc_obj.T = self.parameters[i]["temperature"]
                self.mc_obj.chemical_potential = \
                    self.parameters[i]["chemical_potential"]
            else:
                msg = "Parameter sweep for the MC object not supported yet!"
                raise NotImplementedError(msg)
            self.mc_obj.runMC(steps=self.nsteps, equil=True,
                              equil_params=self.equil_params)

            data = self.mc_obj.get_thermodynamic()
            if self.data_getter is not None:
                # Default get the thermodynamic quantities
                additional_data = self.data_getter(self.mc_obj)
                for key, value in additional_data:
                    data[key] = value
            all_data.append(data)

        if self.outfile != "":
            self.save(all_data)

    def save(self, data):
        """Save data as arrays."""

        data_flattened = {key: [] for key in data[0].keys()}
        for dset in data:
            for key, value in dset.items():
                data_flattened[key].append(value)

        data_flattened = {key: np.array(value) for key, value in
                          data_flattened.items()}
        # Append this to the existing files
        with h5.File(self.outfile, 'a') as hf:
            for key, value in data_flattened.items():
                if key in hf:
                    dset = hf[key]
                    current_size = len(np.array(dset))
                    dset.resize((current_size + len(value),))
                    dset[-len(value):] = value
                else:
                    dset = hf.create_dataset(key, data=value, maxshape=(None,))
        print("Data written to {}".format(self.outfile))
