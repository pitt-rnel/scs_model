from builtins import str
from builtins import range
from mpi4py import MPI
import random as rnd
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
sizeComm = comm.Get_size()
seedValFile = "../code/tools/seedVal.txt"

def save_seed(val):
    """ saves seed val. """
    if rank == 0:
        with open(seedValFile, "wb") as f:
            f.write(str(int(val)).encode())

def load_seed():
    """ loads seed val. Called by all scripts that need the shared seed value. """
    seed = None
    if rank == 0:
        with open(seedValFile, "rb") as f:
            seed = int(f.read())
        with open(seedValFile, "wb") as f:
             f.write(str(seed+1).encode())
    seed = comm.bcast(seed,root=0)
    return seed

def set_seed():
    seed = load_seed()
    for i in range(sizeComm):
        if i==rank:
            rnd.seed(seed+rank)
            np.random.seed(seed+rank)
