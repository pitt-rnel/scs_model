import sys
from mpi4py import MPI
import argparse
import time
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
    parser = argparse.ArgumentParser(description="Input parameters for simulation")
    parser.add_argument("inputFile", help="neural network structure file")
    parser.add_argument("time", help="total simulation time", type=float, default=10000)
    parser.add_argument("rounds", help="number of simulations", type=int, default=5)
    parser.add_argument("eesFrequency", help="ees frequency", type=float, default=33.0)
    parser.add_argument("eesAmplitude", help="ees amplitude applied on both pud and pel", type=float, default=300)
    parser.add_argument("Pud", help="pudendal nerve recruitment rate", type=float, default=1.0)
    parser.add_argument("Pel", help="pelvic nerve recruitment rate", type=float, default=0.0)
    parser.add_argument("SPN", help="SPN neuron recruitment rate", type=float, default=0.0)
    parser.add_argument("label", help="instance label for grouping data", type=int, default=0)
    parser.add_argument("start_vol", help="the bladder volume at the start of the simulation", type=float,
                        default=20.0)
    parser.add_argument("end_vol", help="the bladder volume at the end of the simulation", type=float, default=20.0)
    args = parser.parse_args()

    # fetch the stim frequencies, can be edited to include multiple frequencies
    freq_case = [args.eesFrequency]
    # if needs to test multiple frequencies, replace freq_case with
    # freq_case = [10, 33, 50, 100, 200]

    bp_mean = {}  # record the average bladder pressure of pre/post stimulation
    # auc = {}  # record the auc under bladder pressure traces of pre/post stimulation, for ramp filling experiments
    for freq in freq_case:
        bp_mean[freq] = []
        # auc[freq] = []

    # if no recruitment ratio is detected, then calculate the recruitment ratio based on the stim amplitude and recruitment curve
    if args.Pud == 0.0 and args.Pel == 0.0 and args.SPN == 0.0:
        amp = args.eesAmplitude
    else:
        amp = [args.Pud, args.Pel, args.SPN]

    # for each given stim frequency initiate the neural network and calculate
    for freq in freq_case:
        for i in range(args.rounds):
            print("round: " + str(i + 1))  # the current simulation round
            simrun = bladderModel("frwSimCat.txt", freq, amp, args.time, args.label, args.start_vol, args.end_vol)
            simrun.createNetwork()

            # record the average bp/auc of pre/post stimulation during simulation
            bp_mean[freq].append((simrun.bp_pre, simrun.bp_post, simrun.bp_ratio))
            # auc[freq].append((simrun.auc_pre, simrun.auc_post, simrun.auc_ratio))
            
            # record the recruited percentage of each afferent
            recruited_percentage = simrun.percFiberActEes

    try:
        res_file = open('../../results/' + str(args.label) + time.strftime(
            "_%m_%d_%H_%M") + "_bp-%.1f pud-%.1f pel-%.1f spn at %.1f.txt" % (recruited_percentage[1],recruited_percentage[2], recruited_percentage[3], freq),
                        'wt')
        res_file.write(str(bp_mean))
        res_file.close()
        # res_file2 = open('../../results/' + str(args.label) + time.strftime(
        #     "_%m_%d_%H_%M") + "_auc-%.1f pud-%.1f pel-%.1f spn at %.1f.txt" % (args.Pud, args.Pel, args.SPN, freq),
        #                  'wt')
        # res_file2.write(str(auc))
        # res_file2.close()

    except:
        print('unable to write the file')


if __name__ == '__main__':
    """ The entrance for starting the model
    users can define the recruitment rate of nerves, stimulation amplitude and stimulation frequency.    
    """
    main()
