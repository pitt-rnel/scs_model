from __future__ import print_function
from builtins import range
from .Cell import Cell
from neuron import h
import random as rnd
import numpy as np
from mpi4py import MPI
from tools import seed_handler as sh

sh.set_seed()

comm = MPI.COMM_WORLD
sizeComm = comm.Get_size()
rank = comm.Get_rank()


class PMC(Cell):
    """ Model of the PMC: simulate a switch mechanism of


    """
    __updatePeriod = 25  # Time period in ms between calls of the update fcn, was 0.1, set to 25

    def __init__(self, delay): # delay=1
        """ Object initialization.

        Keyword arguments:
        delay -- time delay in ms needed by a spike to travel the whole fiber
        """
        Cell.__init__(self)
        self._debug = False

        self._set_delay(delay)
        self.maxFiringRate = 200
        self._maxSensorySpikesXtime = int(float(self._delay) / 1000. * float(self.maxFiringRate) + 2) # 2
        # Mean refractory period of 1.6 ms - 625 Hz
        noisePerc = 0.1

        self._refractoryPeriod = rnd.normalvariate(1.6, 1.6 * noisePerc)
        if self._refractoryPeriod > 1000. / self.maxFiringRate:
            self._refractoryPeriod = 1000. / self.maxFiringRate
            print("Warning: refractory period longer than period between 2 natural pulses")

        self.initialise()
        self.cell = h.AfferentFiber()

        # Create a netcon to so that the fiber can receive external stimulation and fire in response
        self._fire = h.NetCon(None, self.cell)

        # Boolean flag to record a segment of the fiber over time
        self._record = False

    """
    Specific Methods of this class
    """

    def initialise(self, lastSpikeTime=0):
        """ Initialise the fiber. """

        # Initial firing rate of .1 Hz
        self._interval = 9999.
        # the initial firing rate is 0
        self._oldFr = 0.
        # the last time the fiber naturally spikes
        self._lastNaturalSpikeTime = lastSpikeTime
        self._oldTime = 0.
        # tolerance to check for events
        self._tolerance = self.__class__.__updatePeriod / 10. # 2.5
        # Create list containing the natural spikes
        self._naturalSpikes = [None] * self._maxSensorySpikesXtime # length=2
        # Create list containing the EES induced spikes
        # self._eesSpikes = [None] * self._maxEesSpikesXtime # length=3
        # Last spike in stim position
        # self._lastStimPosSpikeTime = -9999.
        # Stats
        self._nCollisions = 0
        self._nNaturalSent = 0
        self._nNaturalArrived = 0

    # The delay correspond to the value naturalSpikes[] should have before being sent
    def _set_delay(self, delay):
        """ Set the delay.

        Keyword arguments:
        delay -- time delay in ms needed by a spike to travel the whole fiber
        """
        minDelay = 1
        maxDelay = 100
        if delay >= minDelay and delay <= maxDelay:
            self._delay = delay
        else:
            raise Exception("Afferent fiber delay out of limits")

    def set_firing_rate(self, fr, noise=True):
        """ Set the afferent firing rate.

        Keyword arguments:
        fr -- firing rate in Hz
        oldFr: firing rate from last update
        """

        if fr == self._oldFr: return
        # refractory period
        if fr <= 0:
            self._interval = 99999.
        elif fr >= self.maxFiringRate:
            self._interval = 1000.0 / self.maxFiringRate
        elif fr < self.maxFiringRate and noise:
            mean = 1000.0 / fr  # ms
            sigma = mean * 0.2 # changed from 0.2 to 0.15
            self._interval = rnd.normalvariate(mean, sigma)
        else:
            self._interval = 1000.0 / fr  # ms
        self._oldFr = fr

        # Check whether after setting the new fr the fiber is ready to fire
        if (h.t - self._lastNaturalSpikeTime) >= self._interval - (self.__class__.__updatePeriod / 2.):
            # In this case, shift a bit randomly the last natural spike time in order to reduce an
            # artificially induced synchronized activity between the different modeled fibers
            self._lastNaturalSpikeTime = h.t - np.random.uniform(self._interval / 2., self._interval, 1)

    def update(self, time, end_stim):
        if not end_stim:
            self._update_activity(time)
        else:
            self._update_after_stim_ends(time)
        if self._record: self._record_segment(time)


    def _update_activity(self, time):
        dt = time - self._oldTime
        self._oldTime = time
        # clumsy, can be simplified with class
        # Check whether a natural spike arrived to the end of the fiber
        for i in range(len(self._naturalSpikes)):
            # if self._naturalSpikes[i]) != None:
            if self._naturalSpikes[i] is not None and self._naturalSpikes[i] >= self._delay - self._tolerance:
                # print("there is natural spike!")
                # if self._naturalSpikes[i] >= self._delay - self._tolerance:
                self._fire.event(time, 1)
                self._naturalSpikes[i] = None
                self._nNaturalArrived += 1  # for statistics
                if self._debug: print("\t\t\tnatural spike arrived at time: %f" % (time))

        # update _naturalSpikes
        for i in range(len(self._naturalSpikes)):
            if self._naturalSpikes[i] == None: continue
            # check for collision
            for j in range(len(self._eesSpikes)):
                if self._eesSpikes[j] == None: continue
                if self._naturalSpikes[i] + self.__class__.__updatePeriod > self._eesSpikes[j] - self._tolerance \
                        or self._naturalSpikes[i] < self._eesSpikes[j] + self._tolerance:
                    self._naturalSpikes[i] = None
                    self._eesSpikes[j] = None
                    self._nCollisions += 1
                    if self._debug: print("\t\t\t\tantidromic collision occured at time: %f" % (time))
                    break
            if self._naturalSpikes[i] != None:
                self._naturalSpikes[i] += dt
                if self._naturalSpikes[i] > self._stimPosition - self._tolerance and self._naturalSpikes[
                    i] < self._stimPosition + self._tolerance:
                    if time - self._lastStimPosSpikeTime <= self._refractoryPeriod:
                        self._naturalSpikes[i] = None
                    else:
                        self._lastStimPosSpikeTime = time

        # check for new AP
        if (time - self._lastNaturalSpikeTime) >= self._interval - (self.__class__.__updatePeriod / 2.):
            self._lastNaturalSpikeTime = time
            for i in range(len(self._naturalSpikes)):
                if self._naturalSpikes[i] == None:
                    self._naturalSpikes[i] = 0
                    self._nNaturalSent += 1
                    if self._debug: print("\tsensory spike generated at time: %f" % (time))


    def _update_after_stim_ends(self, time):
        dt = time - self._oldTime
        self._oldTime = time
        # clumsy, can be simplified with class
        # Check whether a natural spike arrived to the end of the fiber
        for i in range(len(self._naturalSpikes)):
            # if self._naturalSpikes[i]) != None:
            if self._naturalSpikes[i] is not None and self._naturalSpikes[i] >= self._delay - self._tolerance:
                # print("there is natural spike!")
                # if self._naturalSpikes[i] >= self._delay - self._tolerance:
                self._fire.event(time, 1)
                self._naturalSpikes[i] = None
                self._nNaturalArrived += 1  # for statistics
                if self._debug: print("\t\t\tnatural spike arrived at time: %f" % (time))

        # update _naturalSpikes
        for i in range(len(self._naturalSpikes)):
            if self._naturalSpikes[i] == None: continue
            # check for collision
            for j in range(len(self._eesSpikes)):
                if self._eesSpikes[j] == None: continue
                if self._naturalSpikes[i] + self.__class__.__updatePeriod > self._eesSpikes[j] - self._tolerance \
                        or self._naturalSpikes[i] < self._eesSpikes[j] + self._tolerance:
                    self._naturalSpikes[i] = None
                    self._eesSpikes[j] = None
                    self._nCollisions += 1
                    if self._debug: print("\t\t\t\tantidromic collision occured at time: %f" % (time))
                    break
            if self._naturalSpikes[i] != None:
                self._naturalSpikes[i] += dt
                if self._naturalSpikes[i] > self._stimPosition - self._tolerance and self._naturalSpikes[
                    i] < self._stimPosition + self._tolerance:
                    if time - self._lastStimPosSpikeTime <= self._refractoryPeriod:
                        self._naturalSpikes[i] = None
                    else:
                        self._lastStimPosSpikeTime = time

        # check for new AP
        if (time - self._lastNaturalSpikeTime) >= self._interval - (self.__class__.__updatePeriod / 2.):
            self._lastNaturalSpikeTime = time
            for i in range(len(self._naturalSpikes)):
                if self._naturalSpikes[i] == None:
                    self._naturalSpikes[i] = 0
                    self._nNaturalSent += 1
                    if self._debug: print("\tsensory spike generated at time: %f" % (time))

    # def get_delay(self):
    #     """ Return the time delay in ms needed by a spike to travel the whole fiber. """
    #     return self._delay

    # def get_natural_spike_list(self):
    #     return self._naturalSpikes

    # def get_ees_spike_list(self):
    #     return self._eesSpikes

    # def set_recording(self, flag, segment):
    #     """ Set the recording flag and segment.
    #     This is used to record the affernt natural and ees-induceddelay
    #
    #     APs in one fiber segment.
    #
    #     Keyword arguments:
    #     segment -- fiber segment to record (between 0 and fibers delay)
    #     time -- current simulation time
    #     """
    #     if segment > self._delay: raise Exception("Segment to record out of limits")
    #     self._record = flag
    #     self._segmentToRecord = segment
    #     self._trigger = []
    #     self._naturalSignals = []
    #     self._eesInducedSignals = []
    #     self._time = []

    # def _record_segment(self, time):
    #     """ Record the fiber segment.  """
    #     if np.any(
    #             np.isclose(0, np.array(self._naturalSpikes, dtype=np.float), rtol=self.__class__.__updatePeriod / 4.)):
    #         self._trigger.append(1)
    #     else:
    #         self._trigger.append(0)
    #
    #     if np.any(np.isclose(self._segmentToRecord, np.array(self._naturalSpikes, dtype=np.float),
    #                          rtol=self.__class__.__updatePeriod / 4.)):
    #         self._naturalSignals.append(1)
    #     else:
    #         self._naturalSignals.append(0)
    #
    #     if np.any(np.isclose(self._segmentToRecord, np.array(self._eesSpikes, dtype=np.float),
    #                          rtol=self.__class__.__updatePeriod / 4.)):
    #         self._eesInducedSignals.append(1)
    #     else:
    #         self._eesInducedSignals.append(0)
    #
    #     self._time.append(time)

    # def get_recording(self):
    #     """ Get the recorded signal """
    #     if self._record:
    #         return self._naturalSignals, self._eesInducedSignals, self._trigger, self._time
    #     else:
    #         return None, None, None, None

    # def get_stats(self):
    #     """ Return a touple containing statistics of the fiber after a simulation is performed. """
    #     if float(self._nNaturalArrived + self._nCollisions) == 0:
    #         percErasedAp = 0
    #     else:
    #         percErasedAp = float(100 * self._nCollisions) / float(self._nNaturalArrived + self._nCollisions)
    #     return self._nNaturalSent, self._nNaturalArrived, self._nCollisions, percErasedAp

    # @classmethod
    # def get_update_period(cls):
    #     """ Return the time period between calls of the update fcn. """
    #     return AfferentFiber.__updatePeriod
    #
    # @classmethod
    # def get_ees_weight(cls):
    #     """ Return the weight of a connection between an ees object and this cell. """
    #     return AfferentFiber.__eesWeight
    #
    # @classmethod
    # def get_max_ees_frequency(cls):
    #     """ Return the weight of a connection between an ees object and this cell. """
    #     return AfferentFiber.__maxEesFrequency

