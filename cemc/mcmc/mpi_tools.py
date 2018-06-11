import numpy as np

def set_seeds(comm):
    """
    This function guaranties different seeds on different processors
    """
    if ( comm is None ):
        return

    rank = comm.Get_rank()
    size = comm.Get_size()
    maxint = np.iinfo(np.int32).max
    if ( rank == 0 ):
        seed = []
        for i in range(size):
            new_seed = np.random.randint(0,high=maxint)
            while( new_seed in seed ):
                new_seed = np.random.randint(0,high=maxint)
            seed.append( new_seed )
    else:
        seed = None

    # Scatter the seeds to the other processes
    seed = comm.scatter(seed, root=0)

    # Update the seed
    np.random.seed(seed)

    if ( size > 1 ):
        # Verify that numpy rand produces different result on the processors
        random_test = np.random.randint( low=0, high=100, size=100 )
        sum_all = np.zeros_like(random_test)
        comm.Allreduce( random_test, sum_all )
        if ( np.allclose(sum_all,size*random_test) ):
            raise RuntimeError( "The seeding does not appear to have any effect on Numpy's rand functions!" )
