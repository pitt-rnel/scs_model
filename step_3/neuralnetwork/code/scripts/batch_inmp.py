import sys
from mpi4py import MPI
import argparse
import time
import numpy as np
import pathlib

sys.path.append('../code')
from tools import seed_handler as sh

sh.set_seed()
sys.path.append('../nnStructures')
from bladderModel import bladderModel

comm = MPI.COMM_WORLD
sizeComm = comm.Get_size()
rank = comm.Get_rank()
print(MPI.UNIVERSE_SIZE)


def main():

    """
    The entrance of the whole model.
    Receive parameters from command line, initialize neural network.
    Record the bladder pressure traces, neuron firing during simulation.
    Generates the npz files of neuron firing and txt files of bladder pressure change pre/post stimulation
    """
    freqs = [3, 10, 33]
    bp_mean = {}
    pud, pel = 1.0, 0.0

    for freq in freqs:
        bp_mean[freq] = []
        # directory name
        dir_name = "230818/sensitivity/inmp/" + str(freq) + '/0.015/'
        # recruitment amplitude
        amp = [pud, pel, 0.0]
        for i in range(1):
            print("round: " + str(i + 1))  # the current simulation round
            """
            initialize bladderModel class
            args: 
            nn-structure, stim_freq, recruitment amplitude(ratio)
            total_simTime, sim_starts, stim_ends (ms)
            instance_id, start_volume, ending_volume, save_to_directory
            """
            simrun = bladderModel("frwSimCat-inmp.txt", freq, amp, 6000, 3000, 6000, 1, 15.0, 15.0, dir_name)
            simrun.createNetwork()

            # record the average bp/auc of pre/post stimulation during simulation
            bp_mean[freq].append((simrun.bp_pre, simrun.bp_post, simrun.bp_ratio))

        try:
            # example directory: results/stim_freq/pud-pel/filename
            res_file = open('../../results/' + dir_name + "bp-%.1f pud-%.1f pel at %.1f.txt" % (pud, pel, freq), 'wt')
            res_file.write(str(bp_mean))
            res_file.close()


        except:
            print('unable to write the file')


if __name__ == '__main__':
    """ The entrance for starting the model
    users can define the recruitment rate of nerves, stimulation amplitude and stimulation frequency.    
    """
    main()
    print('\a')