from builtins import zip
from builtins import str
from builtins import range
from builtins import object
from mpi4py import MPI
import sys
sys.path.append('../code')
from neuron import h
from cells import Motoneuron
from cells import IntFire
from cells import InMn
from cells import Pel
from cells import Pud
from cells import PMC
import random as rnd
import numpy as np
from tools import seed_handler as sh

sh.set_seed()

comm = MPI.COMM_WORLD
sizeComm = comm.Get_size()
rank = comm.Get_rank()


class NeuralNetwork(object):
    """ Spiking neural network model.

    Model of a spiking neural network that can be built in parallel hosts using MPI.
    """

    def __init__(self, parallelContext, inputFile):
        """ Object initialization.

        Keyword arguments:
        parallelContext -- Neuron parallelContext object.
        inputFile -- txt file specifying the neural network structure.
        """

        self._pc = parallelContext
        if rank == 0: self._inputFile = inputFile

        # Initialize the flags to decide which neurons to record.
        self.recordMotoneurons = True
        self.recordAfferents = True
        self.recordIntFire = True


        # Initialize the lists containing the names of the different types of cells.
        self._motoneuronsNames = []
        self._afferentsNames = []
        self._interNeuronsNames = []

        self._connections = []
        # Build the neural network
        self._read()
        self._init_dictionaries()
        self._create_cells()
        self._create_common_connections()


    def __del__(self):
        """ Object destruction and clean all gid references. """
        self._pc.gid_clear()

    def _read(self):
        """ Define the neural network structure from the input file. """
        self._infoMuscles = []
        self._infoCommonCellsInMuscles = []
        self._infoSpecialCells = []
        self._infoCommonMuscleConnections = []
        self._infoInterMuscSensorimotorConnections = {}
        self._infoSpecialConnections = []

        if rank == 0:
            section = None
            for line in open('../nnStructures/'+self._inputFile, "r"):
                if line[0] == "#" or line[0] == "\n":
                    continue
                elif line[0] == "@":
                    section = float(line[1])
                elif section == 1:
                    self._infoMuscles.append(line.strip("\n").split())
                elif section == 2:
                    self._infoCommonCellsInMuscles.append(line.strip("\n").split())
                elif section == 3:
                    self._infoSpecialCells.append(line.strip("\n").split())
                elif section == 4:
                    self._infoCommonMuscleConnections.append(line.strip("\n").split())

        self._infoMuscles = comm.bcast(self._infoMuscles, root=0)
        self._infoCommonCellsInMuscles = comm.bcast(self._infoCommonCellsInMuscles, root=0)
        self._infoSpecialCells = comm.bcast(self._infoSpecialCells, root=0)
        self._infoCommonMuscleConnections = comm.bcast(self._infoCommonMuscleConnections, root=0)

    def _init_dictionaries(self):
        """ Initialize all the dictionaries contatining cells, cell ids and the recorded action potentials. """
        self.actionPotentials = {}
        self.cellsId = {}
        self.cells = {}

        self._nMuscles = len(self._infoMuscles)
        for muscle, muscAfferentDelay in self._infoMuscles:
            # Create sub-dictionaries for all DoF
            self.actionPotentials[muscle] = {}
            self.cellsId[muscle] = {}
            self.cells[muscle] = {}
            for cellInfo in self._infoCommonCellsInMuscles:
                # add lists containing cell ids/cells/ap
                cellClass = cellInfo[0]
                cellName = cellInfo[1]
                self.cellsId[muscle][cellName] = []
                self.cells[muscle][cellName] = []
                if (cellClass == "Motoneuron") and self.recordMotoneurons:
                    self.actionPotentials[muscle][cellName] = []
                elif cellClass in ["Pud", "Pel", "PMC"] and self.recordAfferents:
                    self.actionPotentials[muscle][cellName] = []
                elif cellClass in ["IntFire", "InMn"] and self.recordIntFire:
                    self.actionPotentials[muscle][cellName] = []

        # Add special cells (specific for some muscles or not muscle related)
        for cellInfo in self._infoSpecialCells:
            groupOrMuscle = cellInfo[0]
            cellClass = cellInfo[1]
            cellName = cellInfo[2]
            if not groupOrMuscle in list(self.cellsId.keys()):
                self.actionPotentials[groupOrMuscle] = {}
                self.cellsId[groupOrMuscle] = {}
                self.cells[groupOrMuscle] = {}

            self.cellsId[groupOrMuscle][cellName] = []
            self.cells[groupOrMuscle][cellName] = []
            if (cellClass == "Motoneuron") and self.recordMotoneurons:
                self.actionPotentials[groupOrMuscle][cellName] = []
            elif cellClass in ["Pel", "Pud", "PMC"] and self.recordAfferents:
                self.actionPotentials[groupOrMuscle][cellName] = []
            elif cellClass in ["IntFire", "InMn"] and self.recordIntFire:
                self.actionPotentials[groupOrMuscle][cellName] = []

    def _create_cells(self):
        """ Create the desired cells and assign them a unique cell Id. """
        cellId = 0
        # Iterate over all dictionaries
        for muscle, muscAfferentDelay in self._infoMuscles:
            for cellInfo in self._infoCommonCellsInMuscles:
                cellClass = cellInfo[0]
                cellName = cellInfo[1]
                cellNumber = cellInfo[2]
                if len(cellInfo) >= 4:
                    neuronParam = cellInfo[3]
                else:
                    neuronParam = None
                cellId = self._create_cell_population(cellId, muscle, muscAfferentDelay, cellClass, cellName,
                                                      cellNumber, neuronParam)
        # Add special cells
        for cellInfo in self._infoSpecialCells:
            groupOrMuscle = cellInfo[0]
            cellClass = cellInfo[1]
            cellName = cellInfo[2]
            cellNumber = cellInfo[3]
            if len(cellInfo) >= 5:
                neuronParam = cellInfo[4]
            else:
                neuronParam = None
            muscAfferentDelay = None
            cellId = self._create_cell_population(cellId, groupOrMuscle, muscAfferentDelay, cellClass, cellName,
                                                  cellNumber, neuronParam)

    def _create_cell_population(self, cellId, muscle, muscAfferentDelay, cellClass, cellName, cellNumber,
                                neuronParam=None):
        """ Create cells populations. """
        for n in range(int(cellNumber)):
            # Lets divide equally the cells between the different hosts
            if n % sizeComm == rank:
                # Assign a cellId to the new cell
                self.cellsId[muscle][cellName].append(cellId)
                # Tell this host it has this cellId
                self._pc.set_gid2node(cellId, rank)
                if cellClass == "Motoneuron":
                    # List containing all realistic motoneurons names
                    if not cellName in self._motoneuronsNames:
                        self._motoneuronsNames.append(cellName)
                    drug = False
                    if neuronParam == "drug":
                        drug = True
                    self.cells[muscle][cellName].append(Motoneuron(drug))
                elif cellClass == "Pud":
                    if not cellName in self._afferentsNames:
                        self._afferentsNames.append(cellName)
                    if neuronParam is not None:
                        delay = int(neuronParam)
                    elif muscAfferentDelay is not None:
                        delay = int(muscAfferentDelay)
                    else:
                        raise Exception("Please specify the afferent fiber delay")
                    self.cells[muscle][cellName].append(Pud(delay))
                elif cellClass == "Pel":
                    if not cellName in self._afferentsNames:
                        self._afferentsNames.append(cellName)
                    if neuronParam is not None:
                        delay = int(neuronParam)
                    elif muscAfferentDelay is not None:
                        delay = int(muscAfferentDelay)
                    else:
                        raise Exception("Please specify the afferent fiber delay")
                    self.cells[muscle][cellName].append(Pel(delay))
                elif cellClass == "PMC":
                    if not cellName in self._afferentsNames:
                        self._afferentsNames.append(cellName)
                    if neuronParam is not None:
                        delay = int(neuronParam)
                    elif muscAfferentDelay is not None:
                        delay = int(muscAfferentDelay)
                    else:
                        raise Exception("Please specify the afferent fiber delay")
                    self.cells[muscle][cellName].append(PMC(delay))
                # check for interneurons intfire 4
                elif cellClass == "IntFire":
                    # List containing all interneurons names
                    if not cellName in self._interNeuronsNames: self._interNeuronsNames.append(cellName)
                    self.cells[muscle][cellName].append(IntFire())
                elif cellClass == "InMn":
                    if not cellName in self._interNeuronsNames: self._interNeuronsNames.append(cellName)
                    self.cells[muscle][cellName].append(InMn())
                else:
                    raise Exception("Unknown cell in the network instructions.... (" + str(cellClass) + ")")
                # Associate the cell with this host and id, the nc is also necessary to use this cell as a source for all other hosts
                nc = self.cells[muscle][cellName][-1].connect_to_target(None)
                self._pc.cell(cellId, nc)

                # Record cells APs
                if (cellClass == "Motoneuron") and self.recordMotoneurons:
                    self.actionPotentials[muscle][cellName].append(h.Vector())
                    nc.record(self.actionPotentials[muscle][cellName][-1])
                elif cellClass in ["Pel", "Pud", "PMC"] and self.recordAfferents:
                    self.actionPotentials[muscle][cellName].append(h.Vector())
                    nc.record(self.actionPotentials[muscle][cellName][-1])
                elif cellClass in ["IntFire", "InMn"] and self.recordIntFire:
                    self.actionPotentials[muscle][cellName].append(h.Vector())
                    nc.record(self.actionPotentials[muscle][cellName][-1])
            cellId += 1
        return cellId

    def _connect(self, sourcesId, targetsId, conRatio, conNum, conWeight, synType, conDelay=0.5):
        """ Connect source cells to target cells.

        Keyword arguments:
        sourcesId -- List with the id of the source cells.
        targetsId -- List with the id of the target cells.
        conRatio -- Define how the source cells are connected to the target cells;
        It can be either "unique"  or "random". With "unique" every source cell is connected
        to every target cell, while with "random" every target cell is connected to n=conNum
        randomly selected source cells.
        conNum -- Number of source cells connected to every target cell. Note that with "unique"
        conRation this parameter is still mandatory for double checking.
        conWeight -- Connection weight.
        synType -- Type of synapse that form the connection. It could be either "artificial" for
        artificial cells or "excitatory"/"inhibitory" for realistic cell models.
        conDelay -- Delay of the synapse in ms (default = 1).
        """
        noisePerc = 0.1
        for targetId in targetsId:
            # check whether this id is associated with a cell in this host
            if not self._pc.gid_exists(targetId):
                continue
            if conRatio == "unique" and len(sourcesId) != conNum:
                raise Exception(
                    "Wrong connections number parameter. If the synapses ratio is 'unique' the number of synapses has to be the same as the number of source cells")
            # retrieve the target for artificial cells
            if synType == "artificial":
                target = self._pc.gid2cell(targetId)
            # retrieve the cell for realistic cells
            elif synType == "excitatory" or synType == "inhibitory":
                cell = self._pc.gid2cell(targetId)
            else:
                raise Exception("Wrong synType")

            # create the connections
            for i in range(conNum):
                # create the target for realistic cells
                if synType == "excitatory" or synType == "inhibitory":
                     target = cell.create_synapse(synType)

                if conRatio == "unique":
                    source = sourcesId[i]
                elif conRatio == "random":
                    source = rnd.choice(sourcesId)
                else:
                    raise Exception("Wrong connections ratio parameter")
                nc = self._pc.gid_connect(source, target)
                # print(type(source))
                nc.weight[0] = conWeight
                nc.delay = conDelay + rnd.normalvariate(0.25, 0.25 * noisePerc)
                self._connections.append(nc)
        comm.Barrier()

    def _create_common_connections(self):
        """ Connect network cells within the same degree of freedom. """
        for muscle, muscAfferentDelay in self._infoMuscles:
            for connection in self._infoCommonMuscleConnections:
                # List of source cells ids
                sourcesId = self.cellsId[muscle][connection[0]]
                # gather the sources all together
                sourcesId = comm.gather(sourcesId, root=0)
                if rank == 0: sourcesId = sum(sourcesId, [])
                sourcesId = comm.bcast(sourcesId, root=0)
                # List of taget cells ids
                targetsId = self.cellsId[muscle][connection[1]]
                # Ratio of connection
                conRatio = connection[2]
                # Number of connections
                conNum = int(connection[3])
                # Weight of connections
                conWeight = float(connection[4])
                # Type of synapse
                synType = connection[5]
                # connect sources to targets
                self._connect(sourcesId, targetsId, conRatio, conNum, conWeight, synType)
                print("weight:", conWeight)


    def update_afferents_ap(self, time, end_stim=False):
        """ Update all afferent fibers ation potentials. """
        # Iterate over all dictionaries
        for muscle in self.cells:
            for cellName in self.cells[muscle]:
                if cellName in self._afferentsNames:
                    for cell in self.cells[muscle][cellName]:
                            cell.update(time, end_stim)

    def set_afferents_fr(self, fr):
        """ Set the firing rate of the afferent fibers.

        Keyword arguments:
        fr -- Dictionary with the firing rate in Hz for the different cellNames.
        """
        # Iterate over all dictionaries
        for muscle in self.cells:
            for cellName in self.cells[muscle]:
                if cellName in self._afferentsNames:
                    for cell in self.cells[muscle][cellName]:
                        cell.set_firing_rate(fr[muscle][cellName])

    def initialise_afferents(self):
        """ Initialise cells parameters. """
        # Iterate over all dictionaries
        for muscle in self.cells:
            for cellName in self.cells[muscle]:
                if cellName in self._afferentsNames:
                    for cell in self.cells[muscle][cellName]: cell.initialise()

    def get_nn_infos(self):
        return self._infoMuscles

    def get_afferents_names(self):
        """ Return the afferents name. """
        return self._afferentsNames

    def get_motoneurons_names(self):
        """ Return the motoneurons names. """
        return self._motoneuronsNames

    def get_interneurons_names(self):
        """ Return the inteurons names. """
        return self._interNeuronsNames
