from .Cell import Cell
from neuron import h

class InMn(Cell):
	""" Integrate and Fire cell.

	This class implement and IntFire4 Neuron object.
	Taus are set as in Moraud et al 2016.
	"""

	def __init__(self):
		""" Object initialization. """
		Cell.__init__(self)

		#Create IntFire4
		self.cell = h.IntFire1()
		self.tau = 5
		self.cell.refrac=300


