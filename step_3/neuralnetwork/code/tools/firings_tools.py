from __future__ import division
from builtins import range
from past.utils import old_div
from mpi4py import MPI
import numpy as np
import random as rnd
import time
from . import seed_handler as sh
sh.set_seed()

if not "comm" in locals(): comm = MPI.COMM_WORLD
if not "sizeComm" in locals(): sizeComm = comm.Get_size()
if not "rank" in locals(): rank = comm.Get_rank()

def exctract_firings(apListVector: object, maxTime: object = 0, samplingRate: object = 1000.) -> object:
	""" Extract cells firings from different hosts and return them to host 0.

	Keyword arguments:
		apListVector -- List of action potentials Neuron vectors.
		maxTime -- Maximum time of the simulation in ms (default = 0).
		samplingRate -- Sampling rate of the extracted signal in Hz (default = 1000).
	"""

	#Check whether we need to extract the maxTime or not
	if maxTime is None:
		maxTime = 0
		computeMaxTime = True
	else: computeMaxTime = False

	# extctraing the action potentials time
	nAp = [apVector.size() for apVector in apListVector]
	nCells = len(apListVector)

	if not nAp: maxNap = 0
	else: maxNap = max(nAp)
	maxNap = comm.gather(maxNap,root=0)
	if rank==0: maxNap=max(maxNap)
	maxNap = int(comm.bcast(maxNap,root=0))

	# exctracting the matrix with the ap time (nCells x maxNap)
	if nAp:
		actionPots = -1*np.ones([nCells,maxNap])
		for i,apVector in enumerate(apListVector):
			for ap in range(int(apVector.size())):
				actionPots[i,ap]=apVector.x[ap]
		if actionPots.size>0 and computeMaxTime: maxTime = actionPots.max()
	if computeMaxTime:
		maxTime = comm.gather(maxTime,root=0)
		if rank==0: maxTime=max(maxTime)
		maxTime = comm.bcast(maxTime,root=0)
	# print(nAp)
	# print(apListVector)
	# exctracting the firings matrix of 0 and 1
	dt = 1000./samplingRate
	actionPots = (old_div(actionPots,dt)).astype(int)
	firings = np.zeros([nCells,1+int(old_div(maxTime,dt))])
	if nAp:
		for i in range(nCells):
			indx = actionPots[i,:]>=0
			firings[i,actionPots[i,indx]]=1

	# If MPI: gather all the firings from all the different hosts
	if sizeComm<=1:	return firings
	firingsAll = comm.gather(firings, root=0)
	firings = None
	if rank==0:
		firings = np.concatenate([firingsAll[0],firingsAll[1]])
		for i in range(2,sizeComm):
			firings = np.concatenate([firings,firingsAll[i]])
	return firings



def compute_mean_firing_rate(firings,samplingRate = 1000.):
	""" Return the mean firing rates given the cell firings.

	Keyword arguments:
		firings -- Cell firings, a 2d numpy array (nCells x time).
		samplingRate -- Sampling rate of the extracted signal in Hz (default = 1000).
	"""

	meanFr = None
	if rank ==0:
		interval = old_div(100*samplingRate,1000) #ms
		nCells = firings.shape[0]
		nSamples = firings.shape[1]

		meanFrTemp = np.zeros(nSamples)
		meanFr = np.zeros(nSamples)
		for i in range(int(interval),nSamples):
			totAp = firings[:,i-int(interval):i].sum()
			meanFrTemp[i-int(round(old_div(interval,2)))]=old_div(old_div(totAp,nCells)*samplingRate,interval)

		# Smooth the data with a moving average
		windowSize = int(old_div(25*samplingRate,1000)) #ms
		for i in range(windowSize,nSamples):
			meanFr[i-int(round(old_div(windowSize,2)))] = meanFrTemp[i-windowSize:i].mean()

	return meanFr

def synth_rat_emg( firings,samplingRate = 1000.,delay_ms=2):
	""" Return the EMG activity given the cell firings.

	Keyword arguments:
		firings -- Cell firings, a 2d numpy array (nCells x time).
		samplingRate -- Sampling rate of the extracted signal in Hz (default = 1000).
		delay_ms -- delay in ms between an action potential (AP) and a motor unit
		action potential (MUAP).
	"""
	EMG = None
	if rank==0:
		nCells = firings.shape[0]
		nSamples = firings.shape[1]

		dt = 1000./samplingRate
		delay = int(old_div(delay_ms,dt))

		# MUAP duration between 5-10ms (Day et al 2001) -> 7.5 +-1
		meanLenMUAP = int(7.5/dt)
		stdLenMUAP = int(old_div(1,dt))
		nS = [int(meanLenMUAP+rnd.gauss(0,stdLenMUAP)) for i in range(firings.shape[0])]
		Amp = [abs(1+rnd.gauss(0,0.2)) for i in range(firings.shape[0])]
		EMG = np.zeros(nSamples + max(nS)+delay);
		# create MUAP shape
		for i in range(nCells):
			n40perc = int(nS[i]*0.4)
			n60perc = nS[i]-n40perc
			amplitudeMod = (1-(np.linspace(0,1,nS[i])**2)) * np.concatenate((np.ones(n40perc),old_div(1,np.linspace(1,3,n60perc))))
			logBase = 1.05
			freqMod = old_div(np.log(np.linspace(1,logBase**(4*np.pi),nS[i])),np.log(logBase))
			EMG_unit = Amp[i]*amplitudeMod*np.sin(freqMod);
			for j in range(nSamples):
				if firings[i,j]==1:
					EMG[j+delay:j+delay+nS[i]]=EMG[j+delay:j+delay+nS[i]]+EMG_unit
		EMG = EMG[:nSamples]

	return EMG
