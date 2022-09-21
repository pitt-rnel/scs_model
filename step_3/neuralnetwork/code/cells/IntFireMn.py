from .Cell import Cell
from neuron import h
import random as rnd
import numpy as np
from past.utils import old_div
import time
from tools import seed_handler as sh
sh.set_seed()


class IntFireMn(Cell):
	""" Integrate and Fire cell.

	This class implement and IntFire4 Neuron object.
	Taus are tuned in order to generate epsps lasting approximately 15 ms (Burke 1968)
	and IPSPs lasting 30 ms (McIntyre 2002)
	"""
	__eesWeight = 1.5 # Weight of a connection between an ees object and this cell

	def __init__(self):
		""" Object initialization. """
		Cell.__init__(self)

		# noisePerc = 0.05

		#Create IntFire4
		self.cell = h.IntFireMn()
		self.cell.taue= 0.5
		self.cell.taui1=5
		self.cell.taui2=10
		self.cell.taum = 30
		# self.cell.taum= rnd.normalvariate(6,6*noisePerc)
		# if self.cell.taum <= self.cell.taui2: self.cell.taum = self.cell.taui2 + 0.25
		# self.cell.refrac=rnd.normalvariate(20,20*noisePerc) # mean 50Hz
		self.cell.refrac = 0


	@classmethod
	def get_ees_weight(cls):
		""" Return the weight of a connection between an ees object and this cell. """
		return IntFireMn.__eesWeight
	


