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
import random
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
		modelType -- dictionary containing the model types ('real' or 'artificial')
		tStop -- Time in ms at which the simulation will stop
		"""

        Simulation.__init__(self, parallelContext)

        if rank == 1:
            print("\nWarning: mpi execution in this simulation is not supported and therefore useless.")
            print("Only the results of the first process are considered...\n")

        self.cells = cells
        self.modelType = modelType
        self.tStop = tStop
        self._set_tstop(tStop)
        self.freq = freq
        self.modelType = modelType
        self.update_bladder_interval = h.dt
        self.update_pelvic_interval = h.dt
        self._set_integration_step(h.dt)
        # self.integrate_bladder_window = 4250
        self.integrate_bladder_window = 5000 # change from 4250 to 2500
        self.initial_bladder_vol = start_vol # if start_vol == end_vol: isovolumetric experiment, else ramp-filling experiment
        self.final_bladder_vol = end_vol
        self.filling_vol = self.final_bladder_vol - self.initial_bladder_vol
        self.filling_speed = self.filling_vol/(tStop/self.update_bladder_interval)
        # initialize a container to record the bladder volume change during simulation
        self.bladder_vol = np.array(
            [self.initial_bladder_vol + x * self.filling_speed for x in range(int(tStop / self.update_bladder_interval))])
        # initialize a container to record the bladder pressure change during simulation, set the minimum pressure to 3
        # for background contractions
        self.bladderPressure = np.full(int(tStop / self.update_bladder_interval)+1, np.random.random())
        # self.bladderPressure = np.random.rand(int(tStop / self.update_bladder_interval) +1)*4
        self.cellNum = len(self.cells['Pud'][0]) # number of cells for each population
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
        idx = int(h.t/h.dt)-1
        for cellName in self.cells:
            if self.modelType[cellName] == "real": # SPN neurons
                for cell_list in self.cells[cellName]:
                    for i, c in enumerate(cell_list):
                        # print(c.__dict__)
                        # self._statesM[cellName][i].append(c.node[3](1).v)
                        # self._statesM[cellName][i].append(c.node[-1](0.5).v)
                        self._statesM[cellName][i,idx]= c.node[-1](0.5).v # change to nparray

                        # print(c.node[3](0.5).v)
                        # self._statesM[cellName][i].append(c.soma(0.5).v)
                        # if c.node[3](1).v>-30:
                        if c.node[-1](0.5).v > 30: # count ap at the end of axon, change -30 to 0
                        # if c.soma(0.5).v > -25:
                            # self.spikes[cellName][i].append(1.0)
                            self.spikes[cellName][i,idx]=1.0
                        else:
                            # self.spikes[cellName][i].append(0.0)
                            self.spikes[cellName][i, idx] = 0.0
            elif self.modelType[cellName] == "artificial": # Pud and Pel afferents, PMC neuron
                for cell_list in self.cells[cellName]:
                    # print(cellName, self._statesM[cellName].shape)
                    for j, c in enumerate(cell_list):
                        # self._statesM[cellName][j].append(c.cell.M(0))
                        self._statesM[cellName][j,idx] = c.cell.M(0)
            elif self.modelType[cellName] == "intfire": # other interneurons
                for cell_list in self.cells[cellName]:
                    for i, c in enumerate(cell_list):
                        # self._statesm[cellName][i].append(c.cell.m)
                        # self._statesM[cellName][i].append(c.cell.M(0))
                        self._statesm[cellName][i, idx] = c.cell.m
                        self._statesM[cellName][i, idx] = c.cell.M(0)
                        if c.cell.M(0) > 0.99:
                            # self.spikes[cellName][i].append(1.0)
                            self.spikes[cellName][i, idx] = 1.0
                        else:
                            # self.spikes[cellName][i].append(0.0)
                            self.spikes[cellName][i, idx] = 0.0

    def _updateBladder(self):
        """
        Update bladder pressure based on current SPN firing and bladder filling volume
        """
        idx = int(h.t/h.dt)
        fire_sum = 0
        for each_cell in self.spikes['SPN']:
            if idx < self.integrate_bladder_window:
                fire_sum = np.sum(each_cell[:idx]) * (self.integrate_bladder_window/idx)
            else:
                fire_sum += np.sum(each_cell[idx-self.integrate_bladder_window:idx])
        # set baseline firing = 10
        OUTFIRE = 10 + (10000 / self.integrate_bladder_window) * fire_sum/self.cellNum
        self.outfire.append(OUTFIRE)
        # print(len(self.bladder_vol), idx)
        # calculate bladder pressure based on current bladder volume and SPN firing
        bvol = self.bladder_vol[idx-1]
        vol_contribution = ((1.5 * bvol - 10) > 0) * (1.5 * bvol - 10)
        # newp = max(0,
        #             vol_contribution - 2.075 * (OUTFIRE+10) + 0.1185 * (OUTFIRE+10) ** 2 - 0.00116 * (OUTFIRE+10) ** 3)
        # newp = max(0,
        #            vol_contribution - 2.175 * (OUTFIRE+10) + 0.1185 * (OUTFIRE+10) ** 2 - 0.00119 * (
        #                        OUTFIRE+10) ** 3)
        newp = max(0, vol_contribution - 1.74 * OUTFIRE + 0.077 * OUTFIRE ** 2 - 0.000618 *
                           OUTFIRE ** 3)


        # update bladder pressure
        # unsolved exception: sometimes index 24094 will be skipped during calculation
        # temporary solution: use the value at 24093 to fill
        if idx == 24093:
            self.bladderPressure[24094] = newp
        self.bladderPressure[idx] = newp



    def _updatePelvic(self):
        """
        calculate stim freq for pelvic afferent based on the most recent bladder pressure
        """
        idx = int(h.t/h.dt) -1
        x = self.bladderPressure[idx]
        # calculate the lower bound of pelvic afferent firing rate
        FRlow = -0.8959764 + 0.52*x - 0.003*x** 2+0.000015*x**3
        # define base firing for Pelvic afferents
        pelAf = max(4, FRlow) # change from 5 to 3

        # set firing rate for each pelvic afferent, the refractory period ~ N(1.6, 0.16) ms
        self.pel_fr.append(pelAf)
        for i in range(self.cellNum):
            self.cells['Pel'][0][i].set_firing_rate(pelAf)

    def _updatePMC(self):
        """
        switch on/off PMC based on pelvic firing rate for DEC
        """
        # calculate the pelvic group firing for the latest integration period
        idx = int(h.t /h.dt) - 1
        # print("pel:", self._statesM['Pel'][:,idx].shape)
        fire_sum = np.sum(self._statesM['Pel'][:,idx-10:idx])
        # for each_pel in self._statesM['Pel']:
        #     if each_pel[-1]:
        #         fire_sum += 1
        # switch on/off PMC
        for i in range(self.cellNum):
            if fire_sum > 15:
                self.cells['PMC'][0][i].set_firing_rate(10)
            else:
                self.cells['PMC'][0][i].set_firing_rate(0)

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


    def save_SPNmembrane_potential(self,dirname):
        """
        save SPN membrane potential
        :return:
        """
        if rank==0:
            spn_mem = self._statesM['SPN']
            spn_name = '../../results/' + dirname + '/'+ str(self.label) + '_' + time.strftime("%m_%d_%H_%M_") + '' + str(
                self.initial_bladder_vol) + 'ml_' + str(self.freq) + 'Hz_' + 'SPN_mem.txt'
            f = open(spn_name, 'wt')
            for elem in spn_mem:
                f.write(str(elem) + ' ')
            f.close()


    def save_bp_traces(self, name, data,dirname):
        """
        save data to txt file, mainly used for recording bladder pressure
        """
        if rank==0:
            file_name = str(self.label) + '_' + time.strftime("%m_%d_%H_%M_") + str(name) + '_'+str(self.initial_bladder_vol) +'ml_'+ str(self.freq)
            f = open('../../results/' + dirname + '/' + file_name + ".txt", 'wt')
            for elem in data:
                f.write(str(elem) + ' ')
            f.close()


    def save_spn_outfire(self, name, data, dirname):
        """
        save spn firing rate (s)
        """
        if rank==0:
            file_name = str(self.label) + '_' + time.strftime("%m_%d_%H_%M_") + str(name)
            f = open('../../results/' + dirname + '/' + file_name + ".txt", 'wt')
            for elem in data:
                f.write(str(elem) + ' ')
            f.close()

    def save_pel_sensory_inputs(self, name, data, dirname):
        """
        save spn firing rate (s)
        """
        if rank==0:
            file_name = str(self.label) + '_' + time.strftime("%m_%d_%H_%M_") + str(name)
            f = open('../../results/' + dirname + '/' + file_name + ".txt", 'wt')
            for elem in data:
                f.write(str(elem) + ' ')
            f.close()



    def save_data_to_sparse_matrix(self,dirname,block=True):
        """
        save the membrane potential of SPN, and spikes of all neuron
        components to txt/sparse matrix
        """
        if rank==0:
            # save spn membrane potential
            # save spikes for all afferents, interneurons, SPN
            for name in ["SPN", "FB","IN_Mn", "IN_Mp", "IN_D", "PMC", "Pud", "Pel"]:
                if name in ['Pud', 'Pel', "PMC"]:
                    data = self._statesM[name]
                else:
                    data = self.spikes[name]
                file_name = '../../results/' +dirname+'/'+ str(self.label) + '_' + time.strftime("%m_%d_%H_%M_") + '' +str(self.initial_bladder_vol) + 'ml_'+str(self.freq)+ 'Hz_'+str(name)+'.npz'
                # np_data = np.array(data)
                sparse_matrix = scipy.sparse.csc_matrix(data)
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
        self.spn_rate = []
        self.pel_fr = []
        self._statesm = {}
        self._statesM = {}
        self.spikes = {}
        self.nCells = len(list(self.cells.keys()))
        for cellName in self.cells:
            self._statesm[cellName] = np.zeros((self.cellNum, int(self.tStop/h.dt)))
            self._statesM[cellName] = np.zeros((self.cellNum, int(self.tStop/h.dt)))
            self.spikes[cellName] = np.zeros((self.cellNum, int(self.tStop/h.dt)))
