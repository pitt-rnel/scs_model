from __future__ import division
from __future__ import print_function
from builtins import range
from past.utils import old_div
from mpi4py import MPI
from neuron import h
from scipy.sparse import coo_matrix
from scipy.integrate import simps

from .ForwardSimulation import ForwardSimulation
from .CellsRecording import CellsRecording
from cells import AfferentFiber
import random as rnd
import time
import numpy as np
from tools import general_tools  as gt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tools import firings_tools as tlsf
import pickle
from tools import seed_handler as sh

sh.set_seed()

comm = MPI.COMM_WORLD
sizeComm = comm.Get_size()
rank = comm.Get_rank()


class ForSimSpinalModulation(ForwardSimulation, CellsRecording):
    """ Integration of a NeuralNetwork object over time given an input.
        The simulation results are the cells membrane potential over time.
    """

    def __init__(self, parallelContext, neuralNetwork, cells, modelType, freq, stim_start, stim_end, afferentInput=None, eesObject=None,
                 eesModulation=None, tSimStop=6000, label=0, start_vol=20.0, end_vol=20.0):
        """ Object initialization.

        parallelContext -- Neuron parallelContext object.
        neuralNetwork -- NeuralNetwork object.
        cells -- dict containing cells list (or node lists for real cells) from which we record the membrane potentials.
        modelType -- dictionary containing the model types ('real' or 'artificial') for every
            list of cells in cells.
        afferentInput -- Dictionary of lists for each type of fiber containing the
            fibers firing rate over time and the dt at wich the firing rate is updated.
            If no afferent input is desired use None (default = None).
        eesObject -- EES object connected to the NeuralNetwork, useful for some plotting
            info and mandatory for eesModulation (Default = None).
        eesModulation -- possible dictionary with the following strucuture: {'modulation':
            dictionary containing a	signal of 0 and 1s used to activate/inactivate
            the stimulation for every muscle that we want to modulate (the dictionary
            keys have to be the muscle names used in the neural network structure), 'dt':
            modulation dt}. If no modulation of the EES is intended use None (default = None).
        tStop -- Time in ms at which the simulation will stop (default = 100). In case
            the time is set to -1 the neuralNetwork will be integrated for all the duration
            of the afferentInput. change to 10000 for 10s simulation
        """

        if rank == 1:
            print("\nWarning: mpi execution in this simulation is not supported and therfore useless.")
            print("Only the results of the first process are considered...\n")
        CellsRecording.__init__(self, parallelContext, cells, modelType, freq, tSimStop, label, start_vol, end_vol)
        # changed tstop here and it really detached everything
        ForwardSimulation.__init__(self, parallelContext, neuralNetwork, stim_end, afferentInput, eesObject, eesModulation, tStop=tSimStop)
        h.dt = 0.1
        self._set_integration_step(h.dt)
        self.tSimStop = tSimStop
        self.stim_start = stim_start
        self.stim_end = stim_end
        self.window = 500
        self.update_bladder_interval = h.dt
        self.update_pelvic_interval = h.dt
        self.label = label
        self.ees = eesObject

    def _initialize(self):
        ForwardSimulation._initialize(self)
        CellsRecording._initialize(self)



    def _update(self):
        """ Update simulation parameters. """
        CellsRecording._update(self)
        CellsRecording._updatePelvic(self, self.update_pelvic_interval)
        ForwardSimulation._update(self)

    def _updatebladder(self):
        CellsRecording._updateBladder(self, self.window, self.update_bladder_interval)

    def save_simulation_data(self, name="", title="", dirname="", block=False):
        CellsRecording.save_bp_traces(self, "bp", self.bladderPressure,dirname)
        CellsRecording.save_SPNmembrane_potential(self,dirname)
        CellsRecording.save_data_to_sparse_matrix(self,dirname)

    def plot_membrane_potatial(self, name="", title="", block=False):
        CellsRecording.plot_statesM(self, 'SPN', name, title, block)
        CellsRecording.plot_statesM(self, 'Pud', name, title, block)
        CellsRecording.plot_statesM(self, 'Pel', name, title, block)
        CellsRecording.plot_statesM(self, 'PMC', name, title, block)
        CellsRecording.plot_statesM(self, 'IN_D', name, title, block)
        CellsRecording.plot_statesM(self, 'IN_Mn', name, title, block)
        CellsRecording.plot_statesM(self, 'IN_Mp', name, title, block)
        CellsRecording.plot_statesM(self, 'FB', name, title, block)


    def bladder_pressure_mean(self):
        pre = np.mean(self.bladderPressure[int(self.stim_start / 2 / h.dt):int(self.stim_start / h.dt)])
        stim = np.mean(self.bladderPressure[int(self.stim_start / h.dt):])
        ratio = stim / pre
        return (pre, stim, ratio)

    def bladder_pressure_auc(self):
        datapoints = len(self.bladderPressure) // 2
        pre_area = simps(self.bladderPressure[:datapoints],dx=5)
        post_area = simps(self.bladderPressure[datapoints:], dx=5)
        ratio = post_area / pre_area
        return (pre_area, post_area, ratio)

