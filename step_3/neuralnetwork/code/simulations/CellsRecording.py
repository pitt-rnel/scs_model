from __future__ import print_function
from mpi4py import MPI
from neuron import h
from .Simulation import Simulation
import time
import numpy as np
import scipy.sparse
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
import sys
sys.path.append('../code')
from tools import seed_handler as sh
sh.set_seed()

warnings.filterwarnings("ignore")

comm = MPI.COMM_WORLD
sizeComm = comm.Get_size()
rank = comm.Get_rank()


class CellsRecording(Simulation):
    """ Record cells membrane potential over time. """

    def __init__(self, parallelContext, cells, modelType, freq, tStop, label, start_vol, end_vol):
        """ Object initialization.
		Keyword arguments:
		parallelContext -- Neuron parallelContext object.
		cells -- dict containing lists of the objects we want to record (either all artificial cells or segments
			of real cells).
		modelType -- dictionary containing the model types ('real' or 'artificial') for every
			list of cells in cells.
		tStop -- Time in ms at which the simulation will stop (default = 100).change to 10000 ms = 10 s
		"""

        Simulation.__init__(self, parallelContext)

        if rank == 1:
            print("\nWarning: mpi execution in this simulation is not supported and therfore useless.")
            print("Only the results of the first process are considered...\n")

        self.cells = cells
        self.modelType = modelType
        self._set_tstop(tStop)
        self.freq = freq
        self.modelType = modelType
        self.update_bladder_interval = h.dt
        self.update_pelvic_interval = h.dt
        self._set_integration_step(h.dt)
        # if start_vol == end_vol: isovolumetric experiment. else ramp-filling experiment
        self.initial_bladder_vol = start_vol
        self.final_bladder_vol = end_vol
        self.filling_vol = self.final_bladder_vol - self.initial_bladder_vol
        self.filling_speed = self.filling_vol/(tStop/self.update_bladder_interval)
        # initialize a container to record the bladder volume change during simulation
        self.bladder_vol = np.array([self.initial_bladder_vol + x * self.filling_speed for x in range(int(tStop/h.dt) + 1)])
        # initialize a container to record the bladder pressure change during simulation
        self.bladderPressure = np.array([self.initial_bladder_vol + x * self.filling_speed for x in range(int(tStop/h.dt) + 1)])
        self.cellNum = len(self.cells['Pud'][0]) # number of cells for each component
        self.label = label

    def _initialize(self):
        """
        initialize the simulation object for bladder pressure control
        """
        Simulation._initialize(self)
        self._initialize_states()
        print("The current updating and integration interval is: ", h.dt, " ms")

    def _update(self):
        """
        Update cell properties for each neuron during simulation
        """
        for cellName in self.cells:
            if self.modelType[cellName] == "real": # SPN neuron
                for cell_list in self.cells[cellName]:
                    for i, c in enumerate(cell_list):
                        self._statesM[cellName][i].append(c.soma(0.5).v)
                        if c.soma(0.5).v > -15:
                            self.spikes[cellName][i].append(1.0)
                        else:
                            self.spikes[cellName][i].append(0.0)
            elif self.modelType[cellName] == "artificial": # Pud and Pel afferents
                for cell_list in self.cells[cellName]:
                    for j, c in enumerate(cell_list):
                        self._statesM[cellName][j].append(c.cell.M(0))
            elif self.modelType[cellName] == "intfire": # other interneurons and PMC
                for cell_list in self.cells[cellName]:
                    for i, c in enumerate(cell_list):
                        self._statesm[cellName][i].append(c.cell.m)
                        self._statesM[cellName][i].append(c.cell.M(0))
                        if c.cell.M(0) > 0.99:
                            self.spikes[cellName][i].append(1.0)
                        else:
                            self.spikes[cellName][i].append(0.0)

    def _updateBladder(self, window, update_bladder_interval):
        """
        Update bladder pressure based on current SPN firing and bladder filling volume
        """
        idx = int(h.t / update_bladder_interval)
        fire_sum = 0
        for each_cell in self.spikes['SPN']:
            for c in each_cell[-int(window / update_bladder_interval):]:
                if c:
                    fire_sum += 1
        # calculate SPN average firing rate
        OUTFIRE = (1000 / window) * fire_sum / len(self.spikes['SPN'])
        self.outfire.append(OUTFIRE)
        # bladder volume should never be negative
        newp = max(0, self.bladder_vol[idx]) + (0.00172 * OUTFIRE ** 2 - 0.019 * OUTFIRE+1.05)
        # update bladder pressure
        self.bladderPressure[idx] = newp


    def _updatePelvic(self, update_pelvic_interval):
        """
        calculate stim freq for pelvic afferent based on the most recent bladder pressure
        """
        idx = int(h.t // update_pelvic_interval)
        x = self.bladderPressure[idx]
        # calculate the lower bound of pelvic afferent firing rate
        FRlow = 0.000015 * x ** 3 - 0.0002 * x ** 2 + 0.05924 * x
        if FRlow <= 0:
            pelAf = np.Inf
        else:
            pelAf = FRlow
        # set firing rate for each pelvic afferent, the refractory period ~ N(1.6, 0.16) ms
        for i in range(len(self._statesM['Pel'])):
            self.cells['Pel'][0][i].set_firing_rate(pelAf)


    def plot_statesM(self, neuron, name="", title="", block=True):
        """
        plot the membrane potential for given neuron
        """
        if rank == 0:
            fig = plt.figure(figsize=(16, 30))
            fig.suptitle(title)
            gs = gridspec.GridSpec(len(self._statesM[neuron]), 1)
            gs.update(left=0.1, right=0.95, top=0.9, bottom=0.1, wspace=0.2, hspace=0.2)
            ax = []
            for i in range(len(self._statesM[neuron])):
                ax.append(plt.subplot(gs[i]))
                ax[i].plot(np.linspace(0, self._get_tstop(), len(self._statesM[neuron][0])), self._statesM[neuron][i])
                ax[i].set_ylabel(i)
            ax[-1].set_xlabel('Time (ms)')
            ax[-1].set_title(title)

            fileName = time.strftime("%m_%d_%H_%M_" + str(neuron) + '_mem potential' + ".pdf")
            plt.savefig(self._resultsFolder + fileName, format="pdf", transparent=True)


    def save_SPNmembrane_potential(self):
        """
        save SPN membrane potential
        :return:
        """
        if rank==0:
            spn_mem = self._statesM['SPN']
            spn_name = '../../results/' + str(self.label) + '_' + time.strftime("%m_%d_%H_%M_") + '' + str(
                self.initial_bladder_vol) + 'ml_' + str(self.freq) + 'Hz_' + 'SPN_mem.txt'
            f = open(spn_name, 'wt')
            for elem in spn_mem:
                f.write(str(elem) + ' ')
            f.close()


    def save_bp_traces(self, name, data):
        """
        save data to txt file, mainly used for recording bladder pressure
        """
        if rank==0:
            file_name = str(self.label) + '_' + time.strftime("%m_%d_%H_%M_") + str(name) + '_'+str(self.initial_bladder_vol) +'ml_'+ str(self.freq)
            f = open('../../results/' + file_name + ".txt", 'wt')
            for elem in data:
                f.write(str(elem) + ' ')
            f.close()

    def save_data_to_sparse_matrix(self,block=True):
        """
        save the membrane potential of SPN, and spikes of all neuron
        components to txt/sparse matrix
        """
        if rank==0:
            # save spn membrane potential
            # save spikes for all afferents, interneurons, SPN
            for name in ["SPN", "FB","IN_Mn", "IN_Mp", "IN_D", "PMC", "Pud", "Pel"]:
                if name in ['Pud', 'Pel']:
                    data = self._statesM[name]
                else:
                    data = self.spikes[name]
                file_name = '../../results/' + str(self.label) + '_' + time.strftime("%m_%d_%H_%M_") + '' +str(self.initial_bladder_vol) + 'ml_'+str(self.freq)+ 'Hz_'+str(name)+'.npz'
                np_data = np.array(data)
                sparse_matrix = scipy.sparse.csc_matrix(np_data)
                scipy.sparse.save_npz(file_name, sparse_matrix)



    def _initialize_states(self):
        """
        initialize containers to record m, M, spikes
        self.outfire: record SPN group firing
        self._statesm: record membrane state variable
        self._statesM: record the analytical value of membrane state at current time
        self.spikes: record neuron spike at current time
        """
        self.outfire = []
        self._statesm = {}
        self._statesM = {}
        self.spikes = {}
        self.nCells = len(list(self.cells.keys()))
        for cellName in self.cells:
            self._statesm[cellName] = []
            self._statesM[cellName] = []
            self.spikes[cellName] = []
            for i in range(self.cellNum):
                self._statesm[cellName].append([])
                self._statesM[cellName].append([])
                self.spikes[cellName].append([])
