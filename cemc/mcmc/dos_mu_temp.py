from cemc.mcmc import SGCCompositionFreeEnergy
import h5py as h5
from ase.visualize import view

class FreeEnergyMuTempArray(object):
    def __init__( self, temps=[], mu=[], dos_params={}, fname="free_energy.h5" ):
        self.temps = temps
        self.mu = mu
        self.dos_params = {}
        self.fname = fname
        self.current_dset = 0

    def get_dset_name( self, temp, pot ):
        """
        Generate dataset name
        """
        mu_part = ""
        for key,value in self.mu[0].iteritems():
            mu_part += "{}{}".format(key,int(value*10000))
        dset_name = "{}K{}".format(int(temp),mu_part)
        return dset_name

    def save_dataset( self, limits, hist, T, chem_pot ):
        dsetname = "sim{}".format(self.current_dset)
        self.current_dset += 1
        with h5.File( self.fname, 'a' ) as hf:
            dset = hf.create_dataset( dsetname, data=hist )
            dset.attrs["limits"] = limits
            dset.attrs["temperature"] = T
            for key,value in chem_pot.iteritems():
                dset.attrs[key] = value

    def run( self, sgc_mc_obj, sgc_comp_free_eng, min_num_steps=1000 ):
        for T in self.temps:
            for chem_pot in self.mu:
                print ("Current temperature {}K. Current chemical potential: {}".format(T,chem_pot))
                sgc_mc_obj.reset()
                sgc_comp_free_eng.reset()
                sgc_mc_obj.T = T
                sgc_mc_obj.chemical_potential = chem_pot
                lim, hist = sgc_comp_free_eng.find_dos( sgc_mc_obj, max_rel_unc=2.0, min_num_steps=min_num_steps, rettype="raw_hist" )
                self.save_dataset( lim, hist, T, chem_pot )
        print ("All results stored in {}".format(self.fname))
