import sys
sys.path.append('../code')
from cells import IntFire
from simulations import CellsRecording
import matplotlib.pyplot as plt
from tools import firings_tools as tlsf


class AfferentRecording(CellsRecording):
    def __init__(self, parallelContext, cells, modelType, tStop, afferentFibers):
        CellsRecording.__init__(self, parallelContext, cells, modelType, tStop)
        self.afferentFibers = afferentFibers
        self.actionPotentials = []
        self._nc = []
        for af in self.afferentFibers:
            self._nc.append(af.connect_to_target(None))
            self.actionPotentials.append(h.Vector())
            self._nc[-1].record(self.actionPotentials[-1])

    def _update(self):
        CellsRecording._update(self)
        if h.t % AfferentFiber.get_update_period() < self._get_integration_step():
            for af in self.afferentFibers:
                af.update(h.t)
        if h.t % 100 < self._get_integration_step():
            for af in self.afferentFibers:
                af.set_firing_rate(int(h.t / 10. - 10))

    def _end_integration(self):
        """ Print the total simulation time and extract the results. """
        CellsRecording._end_integration(self)
        self._extract_results()

    def _extract_results(self):
        """ Extract the simulation results. """
        self.firings = tlsf.exctract_firings(self.actionPotentials, self._get_tstop())
