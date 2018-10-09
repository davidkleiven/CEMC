import numpy as np


def set_seeds(comm):
    """
    This function guaranties different seeds on different processors
    """
    if comm is None:
        return

    rank = comm.Get_rank()
    size = comm.Get_size()
    maxint = np.iinfo(np.int32).max
    if rank == 0:
        seed = []
        for i in range(size):
            new_seed = np.random.randint(0, high=maxint)
            while new_seed in seed:
                new_seed = np.random.randint(0, high=maxint)
            seed.append(new_seed)
    else:
        seed = None

    # Scatter the seeds to the other processes
    seed = comm.scatter(seed, root=0)

    # Update the seed
    np.random.seed(seed)

    if size > 1:
        # Verify that numpy rand produces different result on the processors
        random_test = np.random.randint(low=0, high=100, size=100)
        sum_all = np.zeros_like(random_test)
        comm.Allreduce(random_test, sum_all)
        if np.allclose(sum_all, size * random_test):
            msg = "The seeding does not appear to have any effect on Numpy's "
            msg += "rand functions!"
            raise RuntimeError(msg)


def num_processors():
    """Return the number of processors."""
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        num_proc = comm.Get_size()
    except ImportError:
        num_proc = 1
    return num_proc


def mpi_allreduce(msg):
    """Wraps the allreduce method."""
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        value = comm.allreduce(msg)
    except ImportError:
        value = msg
    return value


def mpi_bcast(msg, root=0):
    """Wraps the broadcast method."""
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        value = comm.bcast(msg, root=root)
    except ImportError:
        value = msg
    return value


def mpi_rank():
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
    except ImportError:
        rank = 0
    return rank


def mpi_allgather(msg):
    """Wraps the allgather method."""
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        value = comm.allgather(msg)
    except ImportError:
        value = [msg]
    return value

def mpi_barrier():
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        comm.barrier()
    except ImportError:
        pass


def has_mpi():
    try:
        from mpi4py import MPI
        return True
    except ImportError:
        pass
    return False


def mpi_communicator():
    try:
        from mpi4py import MPI
        return MPI.COMM_WORLD
    except ImportError:
        pass
    return None


def mpi_max():
    try:
        from mpi4py import MPI
        return MPI.MAX
    except ImportError:
        pass
    return None


def mpi_sum():
    try:
        from mpi4py import MPI
        return MPI.SUM
    except ImportError:
        pass
    return None