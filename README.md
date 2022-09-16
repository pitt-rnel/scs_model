# scs_model
Code for computational modeling of spinal cord stimulation to evoke lower urinary tract nerve activity and bladder pressures.

STEPS: 

Step 1: Imaging and Segmentation, DTI


Step 2: Simulate neural recruitment

This step is run from the code 'complex_model_blind.py' and includes building a finite element model (FEM) based on the imaging data provided, simulates the electromagnetic potential evoked in that model during simulation, populates the model with neural trajectories, and simulates actual neural recruitment on those neurons. All of the neural recruitment data is then saved to an output file. 

- more details to be added

Step 3: Simulate bladder function
