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

    def __init__(self, parallelContext, neuralNetwork, cells, modelType, freq, afferentInput=None, eesObject=None,
                 eesModulation=None, tStop=10000, label=0, start_vol=20.0, end_vol=20.0):
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
        CellsRecording.__init__(self, parallelContext, cells, modelType, freq, tStop, label, start_vol, end_vol)
        ForwardSimulation.__init__(self, parallelContext, neuralNetwork, afferentInput, eesObject, eesModulation, tStop)
        # change here. from 0.1 tp 1
        h.dt = 0.1
        self._set_integration_step(h.dt)
        self.tStop = tStop
        self.start_time = self.tStop//2  #
        # window for calculate spn firing freq
        # self.window = self.tStop//20
        # try to change window
        self.window = 500
        self.update_bladder_interval = h.dt
        # self.cells = cells
        # self.cellNum = len(self.cells['Pud'][0])
        # self.modelType = modelType
        # self.freq = freq
        # set pelvic update time interval
        self.update_pelvic_interval = h.dt
        self.label = label

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

    # def plot(self, muscle, cells, name=""):
    #     """ Plot the simulation results. """
    #     if rank == 0:
    #         fig, ax = plt.subplots(figsize=(16, 7))
    #         ax.plot(self._meanFr[muscle][cells])
    #         ax.set_title('Cells mean firing rate')
    #         ax.set_runel(" Mean firing rate (Hz)")
    #         fileName = time.strftime("%Y_%m_%d_CellsRecordingMeanFR_" + name + ".pdf")
    #         plt.savefig(self._resultsFolder + fileName, format="pdf", transparent=True)
    #         CellsRecording.plot(self, name)

    # def raster_plot(self, name="", plot=True):
    #     if rank == 0:
    #         cellsGroups = gt.naive_string_clustering(list(self._statesM.keys()))
    #         # number of cells in each cell type
    #         cellNums = len(self._statesM['Pud'])
    #         for cellNameList in cellsGroups:
    #             for cell_name in cellNameList:
    #                 if cell_name == 'Pud' or cell_name == 'Pel':
    #                     states = self._statesM[cell_name][0]
    #                     for i in range(1, cellNums):
    #                         states = np.vstack((states, self._statesM[cell_name][i]))
    #                 else:
    #                     states = self.spikes[cell_name][0]
    #                     for i in range(1, cellNums):
    #                         states = np.vstack((states, self.spikes[cell_name][i]))
    #                 shape = (len(states), len(states[0]))
    #                 states = coo_matrix(states)
    #                 fig = plt.figure()
    #                 ax = fig.add_subplot(111, facecolor='black')
    #                 ax.plot(states.col, states.row, 's', color='white', ms=0.5)
    #
    #                 ax.set_xlim(0, states.shape[1])
    #                 ax.set_ylim(-1, states.shape[0])
    #                 ax.set_aspect('auto')
    #                 for spine in ax.spines.values():
    #                     spine.set_visible(False)
    #                 ax.invert_yaxis()
    #                 ax.set_title(name, fontsize=10)
    #                 # Move left and bottom spines outward by 10 points
    #                 ax.spines['left'].set_position(('outward', 1))  # both, change 10 to 20
    #                 ax.spines['bottom'].set_position(('outward', 1))
    #                 # Hide the right and top spines
    #                 ax.spines['right'].set_visible(False)
    #                 ax.spines['top'].set_visible(False)
    #                 ax.xaxis.set_ticks_position('bottom')
    #                 tStop = self._get_tstop()
    #                 plt.yticks(fontsize=8)
    #                 plt.xticks(fontsize=8)
    #                 plt.ylabel(cell_name, fontsize=8)
    #                 plt.xlabel('simulation Time (ms)', fontsize=8)
    #
    #                 fileName = time.strftime("%m_%d_%H_%M_raster_plot_" + name + "_" + cell_name + ".pdf")
    #                 plt.savefig(self._resultsFolder + fileName, format="pdf", transparent=False)

    def save_simulation_data(self, name="", title="", block=False):
        CellsRecording.save_bp_traces(self, "bp", self.bladderPressure)
        CellsRecording.save_SPNmembrane_potential(self)
        CellsRecording.save_data_to_sparse_matrix(self)

    def plot_membrane_potatial(self, name="", title="", block=False):
        CellsRecording.plot_statesM(self, 'SPN', name, title, block,)
        CellsRecording.plot_statesM(self, 'IN_D', name, title, block)
        CellsRecording.plot_statesM(self, 'IN_Mn', name, title, block)
        CellsRecording.plot_statesM(self, 'IN_Mp', name, title, block)
        CellsRecording.plot_statesM(self, 'FB', name, title, block)

    # def save_results(self, name):
    #     if rank == 0:
    #         fileName = time.strftime("%Y_%m_%d_FSSM_nSpikes") + name + ".p"
    #         with open(self._resultsFolder + fileName, 'w') as pickle_file:
    #             pickle.dump(self._nSpikes, pickle_file)
    #             pickle.dump(self._nActiveCells, pickle_file)


    def bladder_pressure_mean(self):
        pre = np.mean(self.bladderPressure[int(self.start_time / 2 / h.dt):int(self.start_time / h.dt)])
        stim = np.mean(self.bladderPressure[int(self.start_time / h.dt):])
        ratio = stim / pre
        return (pre, stim, ratio)

    def bladder_pressure_auc(self):
        datapoints = len(self.bladderPressure) // 2
        pre_area = simps(self.bladderPressure[:datapoints],dx=5)
        post_area = simps(self.bladderPressure[datapoints:], dx=5)
        ratio = post_area / pre_area
        return (pre_area, post_area, ratio)

