
class MCParameterSweep(object):
    known_parameters = {
        "Montecarlo":["temperature","composition"],
        "SGCMonteCarlo":["temperature","chemical_potential"]
    }
    def __init__( self, mc_obj, parameters ):
        self.parameters = parameters
        self.mc_obj = mc_obj
        self.check_initialization()

    def check_initialization( self ):
        """
        Check that the parameters are valid
        """
        if ( not self.mc_obj.__name__ in self.known_parameters.keys() ):
            raise ValueError( "The Monte Carlo instance was not recognized. Known MonteCarlo object: {}".format(known_parameters.keys()) )

        allowed_params = known_parameters[self.mc_obj.__name__]
        for param in self.parameters.keys():
            if ( not param in allowed_params ):
                raise ValueError( "Unrecognized parameters. Has to be one of {}".format(allowed_params))
