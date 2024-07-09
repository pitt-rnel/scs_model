from __future__ import print_function
from builtins import range
from .Cell import Cell
from neuron import h
import random as rnd
import time
import numpy as np
from mpi4py import MPI
from tools import seed_handler as sh

sh.set_seed()

comm = MPI.COMM_WORLD
sizeComm = comm.Get_size()
rank = comm.Get_rank()


class Pud(Cell):
    """ Model of the afferent fiber.

    The model integrates the collision of natural spikes with the ones
    induced by epidural electrical stimulation (EES) of the spinal cord.
    In particular the APs induced by the stimulation and the ones induced
    by the sensory organ are added to relative lists containing all APs
    positions in the fiber at the currant time. Every 0.1 ms (__updatePeriod)
    the position of the APs in the fiber has to be updated in order to
    simulate the progation of the APs and the collision of natural and EES
    induced spikes. A refracory period of mean 1.6 ms and std of 0.16 ms is modeled.
    Note that the __updatePeriod can be incresaed in order to speed up the
    simulations. However, by increasing this value we would also lose resolution
    of the refractory period.
    """
    # change update period from 0.1ms to 25 ms to get smoother plot
    __updatePeriod = 25  # Time period in ms between calls of the update fcn, was 0.1
    __eesWeight = -3
    __maxEesFrequency = 200

    def __init__(self, delay):
        """ Object initialization.

        Keyword arguments:
        delay -- time delay in ms needed by a spike to travel the whole fiber
        """

        Cell.__init__(self)
        self._debug = False

        # Initialise cell parameters
        self._set_delay(delay)
        # self.stim_freq = 30
        self.maxFiringRate = 200  # This should be lower than the frequency allowed by the refractory period
        self._maxSensorySpikesXtime = int(float(self._delay) / 1000. * float(self.maxFiringRate) + 2)
        self._maxEesSpikesXtime = int(float(self._delay) / 1000. * float(self.__class__.__maxEesFrequency) + 2)
        # Mean refractory period of 1.6 ms - 625 Hz
        noisePerc = 0.0
        self._refractoryPeriod = rnd.normalvariate(1.6, 1.6 * noisePerc)
        if self._refractoryPeriod > 1000. / self.maxFiringRate:
            self._refractoryPeriod = 1000. / self.maxFiringRate
            print("Warning: refractory period bigger than period between 2 natural pulses")
        # Position along the fiber recruited by the stimulation
        # self._stimPosition = self._delay-0.5
        self._stimPosition = delay - 0.5

        self.initialise()
        # Create an ARTIFICIAL_CELL Neuron mechanism that will be the source of a netCon object.
        # This will be used to comunicate the APs to target cells
        self.cell = h.AfferentFiber()
        '''
        
        # straightly connect netstim to this fiber, rather than ees object
        self.s = h.NetStim()
        self.s.start = 300 # ms
        self.s.interval = 1000.0/self.stim_freq # 30 hz, interval is calculated in ms
        self.s.number = 100
        self.s.noise = 0.0
        '''

        # Create a netcon to make the fiber fire
        self._fire = h.NetCon(None, self.cell)
        # self._fire = h.NetCon(self.s, self.cell)

        # Boolean flag to record a segment of the fiber over time # change to true
        self._record = False

    """
    Specific Methods of this class
    """

    def initialise(self, lastSpikeTime=0):
        """ Initialise the fiber. """

        # Initial firing rate of .1 Hz
        self._interval = 999999.
        # the initial firing rate is 0
        self._oldFr = 0.
        # the last time the fiber naturally spikes
        self._lastNaturalSpikeTime = lastSpikeTime
        self._oldTime = 0.
        # tolerance to check for events
        self._tolerance = self.__class__.__updatePeriod / 10.
        # Create list containing the natural spikes
        self._naturalSpikes = [None] * self._maxSensorySpikesXtime
        # Create list containing the EES induced spikes
        self._eesSpikes = [None] * self._maxEesSpikesXtime
        # Last spike in stim position
        self._lastStimPosSpikeTime = -999999.
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

    def set_firing_rate(self, fr, noise=False):
        """ Set the afferent firing rate.

        Keyword arguments:
        fr -- firing rate in Hz
        change noise to False
        """

        if fr == self._oldFr: return
        # refractory period
        if fr <= 0:
            self._interval = 99999.
        elif fr >= self.maxFiringRate:
            self._interval = 1000.0 / self.maxFiringRate
        elif fr < self.maxFiringRate and noise:
            mean = 1000.0 / fr  # ms
            sigma = mean * 0.2
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
        """ Update the fiber activity induced by the stimulation.

        It first propagates the action pontentials (APs) induced by the stimulation along
        the fiber and then it checks whether a new pulse of stimulation occured.
        In this case an event is sent to all the connected cells at the time = time
        Then, it checks whether a natural AP reached the end of the fiber
        and in this case it sends an event to the connected cells at time = time.
        It then propagates the natural action pontentials (APs) along the fiber
        taking in to account possible collision with EES induced AP.

        Keyword arguments:
        time -- current simulation time, necessary for synaptic connections
        """
        dt = time - self._oldTime
        self._oldTime = time
        # Propagates the ees antidromic action pontentials
        for i in range(len(self._eesSpikes)):
            if self._eesSpikes[i] != None:
                if self._eesSpikes[i] <= -self._refractoryPeriod:
                    self._eesSpikes[i] = None
                    if self._debug: print("\t\tAntidromic spike arrived at origin - refPeriod at time: %f" % (time))
                else:
                    self._eesSpikes[i] -= dt

        # Check whether a new pulse of stimulation occured
        if self.cell.EES == 1:
            self.cell.EES = 0
            # check whether the fiber isn't in refractory period
            if time - self._lastStimPosSpikeTime > self._refractoryPeriod:
                if self._debug: print("\tStimulation pulse occured at time: %f" % (time))
                self._lastStimPosSpikeTime = time
                self._fire.event(time + self._delay - self._stimPosition, 1)
                for i in range(len(self._eesSpikes)):
                    if self._eesSpikes[i] == None:
                        self._eesSpikes[i] = self._stimPosition
                        break  # attention if not found AP of EES is not considered - depends on the size of eesSpikes

        # Check whether a natural spike arrived to the end of the fiber
        for i in range(len(self._naturalSpikes)):
            # if self._naturalSpikes[i]) != None:
            if self._naturalSpikes[i] is not None and self._naturalSpikes[i] >= self._delay - self._tolerance:
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
            # advance natural AP
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
                    break  # attention if not found, AP of EES is not considered - size of naturalSpike


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

    def get_delay(self):
        """ Return the time delay in ms needed by a spike to travel the whole fiber. """
        return self._delay

    def set_recording(self, flag, segment):
        """ Set the recording flag and segment.
        This is used to record the affernt natural and ees-induced
        APs in one fiber segment.

        Keyword arguments:
        segment -- fiber segment to record (between 0 and fibers delay)
        time -- current simulation time
        """
        if segment > self._delay: raise Exception("Segment to record out of limits")
        self._record = flag
        self._segmentToRecord = segment
        self._trigger = []
        self._naturalSignals = []
        self._eesInducedSignals = []
        self._time = []

    def _record_segment(self, time):
        """ Record the fiber segment.  """
        if np.any(
                np.isclose(0, np.array(self._naturalSpikes, dtype=np.float), rtol=self.__class__.__updatePeriod / 4.)):
            self._trigger.append(1)
        else:
            self._trigger.append(0)

        if np.any(np.isclose(self._segmentToRecord, np.array(self._naturalSpikes, dtype=np.float),
                             rtol=self.__class__.__updatePeriod / 4.)):
            self._naturalSignals.append(1)
        else:
            self._naturalSignals.append(0)

        if np.any(np.isclose(self._segmentToRecord, np.array(self._eesSpikes, dtype=np.float),
                             rtol=self.__class__.__updatePeriod / 4.)):
            self._eesInducedSignals.append(1)
        else:
            self._eesInducedSignals.append(0)

        self._time.append(time)

    def get_recording(self):
        """ Get the recorded signal """
        if self._record:
            return self._naturalSignals, self._eesInducedSignals, self._trigger, self._time
        else:
            return None, None, None, None

    def get_stats(self):
        """ Return a touple containing statistics of the fiber after a simulation is performed. """
        if float(self._nNaturalArrived + self._nCollisions) == 0:
            percErasedAp = 0
        else:
            percErasedAp = float(100 * self._nCollisions) / float(self._nNaturalArrived + self._nCollisions)
        return self._nNaturalSent, self._nNaturalArrived, self._nCollisions, percErasedAp

    def is_artificial(self):
        """Return a flag to check whether the cell is an integrate-and-fire or artificial cell."""
        return 1

    @classmethod
    def get_update_period(cls):
        """ Return the time period between calls of the update fcn. """
        return Pud.__updatePeriod

    @classmethod
    def get_ees_weight(cls):
        """ Return the weight of a connection between an ees object and this cell. """
        return Pud.__eesWeight

    @classmethod
    def get_max_ees_frequency(cls):
        """ Return the weight of a connection between an ees object and this cell. """
        return Pud.__maxEesFrequency
