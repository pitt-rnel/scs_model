# scs_model
Code for computational modeling of spinal cord stimulation to evoke lower urinary tract nerve activity and bladder pressures.

STEPS: 

Step 1: Imaging and Segmentation, DTI


Step 2: Simulate neural recruitment

This step is run from the code 'run_complex_model.py' and includes building a finite element model (FEM) based on the imaging data provided, simulates the electromagnetic potential evoked in that model during simulation, populates the model with neural trajectories, and simulates actual neural recruitment on those neurons. All of the neural recruitment data is then saved to an output file. To run this stage, open run_complex_model.py in a version of Sim4Life 7 or higher that includes A-delta neuron modeling capabilities, and in the "runProject" function change template and contact_n to be the name of the electrode file to load and the contact number to use, respectively. Change num_neuron to be the number of neurons per root and type with which to run the simulation. Additionally, pathnames in main() will need to be changed to local directories. 

After results are saved from the neuron simulation, run plot_analysis_scale.ipynb to re-scale these values to the correct ratio of neurons per root and view initial plots of recruitment values for a simulation. This step will also output a series of recruitment curve values in a text file that can be used in Step 3. 

Step 3: Simulate bladder function
