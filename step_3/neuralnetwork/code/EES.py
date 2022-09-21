from __future__ import division
from __future__ import print_function
from builtins import str
from builtins import range
from builtins import object
from past.utils import old_div
from mpi4py import MPI
from neuron import h
import numpy as np
from scipy import interpolate
from cells import Motoneuron
from cells import IntFireMn
from cells import AfferentFiber
import random as rnd
import time
from tools import firings_tools as tlsf
from tools import seed_handler as sh
sh.set_seed()

comm = MPI.COMM_WORLD
sizeComm = comm.Get_size()
rank = comm.Get_rank()

class EES(object):
	""" Epidural Electrical Stimulation model. """

	def __init__(self,parallelContext,neuralNetwork,amplitude,frequency,simTime, pulsesNumber=100000,species="cat"):
		""" Object initialization.

		Keyword arguments:
		parallelContext: Neuron parallelContext object.
		neuralNetwork: NeuralNetwork instance to connect to this object.
		amplitude: Aplitude of stimulation. It could either be an integer value between _minCur and _maxCur or a list containing the percentages
		of recruited primary afferents, secondary afferents and motoneurons.
		frequency: Stimulation frequency in Hz; it has to be lower than the
		maximum stimulation frequency imposed by the AfferentFiber model.
		pulsesNumber: number of pulses to send (default 100000).
		species -- rat or human (it loads different recruitment curves)
		"""
		self._pc = parallelContext
		self._nn = neuralNetwork
		self._species = species
		self._eesId = 1000000
		self._connections = {}
		self.simTime = simTime

		self._maxFrequency = AfferentFiber.get_max_ees_frequency()
		self._current = None
		self._percIf= None # percentage of afferent I to be recruited
		self._percIIf= None
		self._percMn = None

		if rank==0:
			self._pc.set_gid2node(self._eesId, rank)
			self._stim = h.NetStim()
			self._stim.number = pulsesNumber
			self._stim.start = self.simTime//2 # stimulation starts at the middle of simulation
			self._stim.noise = 0

			self._pulses = h.Vector()
			nc = h.NetCon(self._stim,None)
			self._pc.cell(self._eesId, nc)
			nc.record(self._pulses)

			# Load the recruitment data
			self._load_rec_data()

		# EES can be applied on pud, pel and SPN
		self._recruitedCells = ['Pud','Pel','SPN']

		# Initialize neuron connections and apply EES
		self._connect_to_network()
		self.set_amplitude(amplitude)
		self.set_frequency(frequency)

	def __del__(self):
		self._pc.gid_clear()

	def _load_rec_data(self):
		""" Load recruitment data from a previosly validated FEM model (Capogrosso et al 2013). """
		if rank==0:
			if self._species=='cat':
				recI_MG = np.loadtxt('../../inputs/PudAff.txt')
				recII_MG = np.loadtxt('../../inputs/PelAff.txt')
				recMn_MG = np.loadtxt('../../inputs/SPN.txt')
			else:
				raise Exception("Need effective recruitment data.")

			# the range of EES current
			self._minCur = 0 #uA
			self._maxCur = 1000 #uA

			nVal = recI_MG.size
			allPercIf = recI_MG
			allPercIIf = recII_MG
			allPercMn = recMn_MG

			currents = np.linspace(self._minCur,self._maxCur,nVal)
			# calculate recruitment of Pud, Pel, SPN using current
			self._tckIf = interpolate.splrep(currents, allPercIf)
			self._tckIIf = interpolate.splrep(currents, allPercIIf)
			self._tckMn = interpolate.splrep(currents, allPercMn)


	def _connect(self,targetsId,cellType,netconList):
		""" Connect this object to target cells.

		Keyword arguments:
		targetsId -- List with the id of the target cells.
		cellType -- String defining the cell type.
		netconList -- List in which we append the created netCon Neuron objects.
		"""
		delay=1
		if cellType in self._nn.get_afferents_names():
			weight = AfferentFiber.get_ees_weight()
		elif cellType in self._nn.get_real_motoneurons_names():
			weight = Motoneuron.get_ees_weight()
		else: raise Exception("Undefined cell type for EES.")

		for targetId in targetsId:
			if not self._pc.gid_exists(targetId): continue

			if cellType in self._nn.get_real_motoneurons_names():
				cell = self._pc.gid2cell(targetId)
				target = cell.create_synapse('ees')
			else: target = self._pc.gid2cell(targetId)

			nc = self._pc.gid_connect(self._eesId,target)
			nc.weight[0] = weight
			nc.delay = delay
			nc.active(False)
			netconList.append(nc)

	def _connect_to_network(self):
		""" Connect this ees object to the NeuralNetwork object. """
		for muscle in self._nn.cellsId:
			self._connections[muscle] = {}
			for cellType in self._nn.cellsId[muscle]:
				if cellType in self._recruitedCells:
					self._connections[muscle][cellType] = []
					self._connect(self._nn.cellsId[muscle][cellType],cellType,self._connections[muscle][cellType])
				comm.Barrier()

	def _activate_connections(self,netcons,percentage):
		""" Modify which connections are active. """
		for nc in netcons: nc.active(False)
		nCon = comm.gather(len(netcons),root=0)
		nOn = None
		if rank==0:
			nCon = sum(nCon)
			if not percentage:
				percentage = 0
			nOnTot = int(round(percentage*nCon))
			nOn = np.zeros(sizeComm) + old_div(nOnTot,sizeComm)
			for i in range(nOnTot%sizeComm): nOn[i]+=1
		nOn = comm.scatter(nOn, root=0)

		ncIndexes = list(range(len(netcons)))
		rnd.shuffle(ncIndexes)
		for indx in ncIndexes[:int(nOn)]: netcons[indx].active(True)

	def set_amplitude(self,amplitude,muscles=None):
		""" Set the amplitude of stimulation.

		Note that currently all DoFs have the same percentage of afferents recruited.
		Keyword arguments:
		amplitude -- Amplitude of stimulation. It coulde either be an integer
		value between _minCur and _maxCur or a list containing the percentages
		of recruited primary afferents, secondary afferents and motoneurons.
		muscles -- list of muscle names on which the stimulation amplitude is
		modifiel. If no value is specified, none is used and all the amplitude is
		modified on all the network muscles.
		"""

		if rank == 0:
			if isinstance(amplitude,int) or isinstance(amplitude,float):
				if amplitude > self._minCur and amplitude <=self._maxCur:
					self._current = amplitude
					self._percIf=  interpolate.splev(amplitude,self._tckIf)
					if self._percIf<0:self._percIf=0
					self._percIIf=  interpolate.splev(amplitude,self._tckIIf)
					if self._percIIf<0:self._percIIf=0
					self._percMn =  interpolate.splev(amplitude,self._tckMn)
					if self._percMn<0:self._percMn=0
				else:                        
					raise Exception("Current amplitude out of bounds - min = "+str(self._minCur)+"/ max = "+str(self._maxCur))
			# if recruitment rate is given, use percentage instead of current
			elif isinstance(amplitude,list) and len(amplitude)==3:
				self._current = "NA" # change -1 to NA
				self._percIf= amplitude[0]
				self._percIIf= amplitude[1]
				self._percMn = amplitude[2]
			else: raise Exception("badly defined amplitude")
            

		self._current = comm.bcast(self._current,root=0)
		self._percIf = comm.bcast(self._percIf,root=0)
		self._percIIf = comm.bcast(self._percIIf,root=0)
		self._percMn = comm.bcast(self._percMn,root=0)


		if muscles is None: muscles = list(self._nn.cellsId.keys())
		for muscle in muscles:
			for cellType in self._nn.cellsId[muscle]:
				if cellType in self._nn.get_primary_afferents_names():
					self._activate_connections(self._connections[muscle][cellType],self._percIf)
				elif cellType in self._nn.get_secondary_afferents_names():
					self._activate_connections(self._connections[muscle][cellType],self._percIIf)
				elif cellType in self._nn.get_motoneurons_names():
					self._activate_connections(self._connections[muscle][cellType],self._percMn)

	def set_frequency(self,frequency):
		""" Set the frequency of stimulation.

		Note that currently all DoFs have the same percentage of afferents recruited.
		Keyword arguments:
		frequency -- Stimulation frequency in Hz; it has to be lower than the
		maximum stimulation frequency imposed by the AfferentFiber model.
		"""
		if rank == 0:
			if frequency>0 and frequency<self._maxFrequency:
				self._frequency = frequency
				self._stim.interval = 1000.0/self._frequency
			elif frequency<=0:
				self._frequency = 0
				self._stim.interval = 100 * self.simTime
			elif frequency>=self._maxFrequency:
				raise Exception

	def get_amplitude(self,printFlag=False):
		""" Return the stimulation amplitude and print it to screen.

		If set_amplitude was used with the non default 'muscles' parameter,
		the stimulation amplitude here returned is not valid for the whole network.
		Indeed, this function only returns the most recent amplitude value that was used
		to change the stimulation settings. """

		if rank==0 and printFlag:
			print("The stimulation amplitude is set at: "+str(self._current)+" uA")
			print("\t"+str(int(self._percIf*100))+"% of Pudendal nerve recruited")
			print("\t" + str(int(self._percIIf * 100)) + "% of Pelvic nerve recruited")
			print("\t" + str(int(self._percMn * 100)) + "% of SPN neuron recruited")
		return self._current, self._percIf, self._percIIf, self._percMn

	def get_frequency(self,printFlag=False):
		""" Return the stimulation frequency and print it to screen. """
		frequency = None
		if rank==0:
			if printFlag: print("The stimulation frequency is set at: "+str(self._frequency)+" Hz")
			frequency = int(round(1000./self._stim.interval))
		frequency = comm.bcast(frequency,root=0)
		return frequency

	def get_n_pulses(self):
		""" Return the number of pulses to send at each burst. """
		nPulses = None
		if rank==0: nPulses = self._stim.number
		nPulses = comm.bcast(nPulses,root=0)
		return nPulses

	def get_id(self):
		""" Return the ID of the NetStim object. """
		return self._eesId

	def get_pulses(self,tStop,samplingRate=1000.):
		""" Return the stimulation pulses. """
		if rank==0:
			nPulses = self._pulses.size()
			if not nPulses: return None,None
			pulsesTime = np.array([self._pulses.x[pulseInd] for pulseInd in range(int(nPulses))])
			dt = 1000./samplingRate
			pulsesTime = (old_div(pulsesTime,dt)).astype(int)
			pulses = np.zeros(1+int(old_div(tStop,dt)))
			pulses[pulsesTime]=1
			return pulsesTime,pulses
		return None,None
