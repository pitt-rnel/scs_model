# scs_model
Code for computational modeling of spinal cord stimulation to evoke lower urinary tract nerve activity and bladder pressures.

## STEPS: 

## Step 1: Image segmentation and tractography

### 1.1 Automatic segmentation
This step is run from the code '****' and performs automatic segmentation of the spinal cord tissues (gray matter, white matter, dorsal/ventral roots and rootlets, cerebrospinal fluid, and epidural fat) in the MR dataset using a pre-trained Deep Neural Network. It generates and saves the segmentation masks, Jaccard Index and Hausdorff Distance metrics into the output folder. This code may be run with any python interpreter. Inspection and correction of the segmentation can be performed with any image processing software, such as iSeg or FSL.

**Input:** MR dataset (.nii)

**Output:** Automatically segmented images (.nii), Jaccard index & Hausdorff distance (.txt)

**System Requirements:**
need to add

**Instructions:** The user needs to upload a zipped directory containing the following folders: 
1. The user should create an **"input"** folder that contains the acquired imaging data with file with names **19f, t2** and **fat**. These files can either be in the .nii or the .nii.gz format. Note that the input dimension of these images is fixed (see below).

2. The user should crease an **"output"** folder that contains a few manually segmented examples for each tissue to be segmented. This is to allow masking of the spinal cord for better performance. The files should have the following names:
csf.nii.gz, dorsalroots.nii.gz, epiduralfat.nii.gz, graymatter.nii.gz, ventralroots.nii.gz, whitematter.nii.gz

3. Run the code '****' from the folder containing the "input" and "output" folders.

4. Check the segmentation results.

The default Neural Network (hybrid cat1-cat3) used in this node was trained with manual segmentations of 2 cat lumbosacral spine samples distributed along the cord (cat spine 1: 207 slices & cat spine 3: 176 slices). Note that to use this network, the raw images have to be of 384 x 512 pixel dimension in the axial plane.

![Asset 3](https://user-images.githubusercontent.com/43448105/191312158-00045d2b-d32c-4bdf-8836-307683e137f6.png)

**Limitations**: This pipeline currently does not support training new weights for the Neural Network. If the user has data that is substantially different from the cat lumbosacral spine (ex. cat cervical spine) and simply running the default does not give satisfactory segmentations, the user may upload a Neural Network pre-trained with manual segmentation of the user’s data. To generate this pre-trained Network in the correct .pickle format, contact Alessandro Fasse (fasse@zmt.swiss).

### 1.2 Generate neuron axon trajectories
This step does not require a code, but is necessary to perform to generate neuron trajectories for functionalizing the spinal cord model. We suggest using [DSI_Studio](https://dsi-studio.labsolver.org/), a tractography software tool for diffusion MRI processing. It allows construction of realistic axonal trajectories based on DTI data, which can later be placed as neurons in the 3D model.

**Input:** 1-DTI images (.nii), 2-b-table (.txt)
**Output:** tracked fiber coordinates (.txt)

**Instructions:** 
1. In Step T1, import DTI image files. You will be prompted to add the b-table.
2. In Step T2, import the .src file generated in Step T1, select DTI checkbox, Check b-table checkbox. Click on Run Reconstruction.
3. In Step T3, import the .fib file generated in Step T2.
4. Tracking:
*Region of Interest (ROI):*
- For each root level, create a new region of interest by selecting Regions -> New region.  
- Draw the region of interest by selecting a desired image slice and drawing around the spinal root. We suggest picking a slice just rostral to the DRG of a target root. Please note that for the default use of the pipeline, the fibers should be tracked individually for each root level (L6, L7 etc.), side(ventral/dorsal) and location (right/left).
**Example:** for spinal roots L6 the user will have to track and save separately the following files: L6_r_d (right, dorsal), L6_l_d (left,dorsal), L6_r_v (right, ventral), L6_l_v (left, ventral).

*Tracking parameters:*
Tracking parameters may differ depending on the length of your specific dataset. We suggest the following values for a 8cm long dataset: Angular Threshold=60, Min Length=20, Max Length=100, Terminate if=10000 Tracts.
Click on Fiber Tracking and visualize the result.

*Tract cleaning:*
To ensure that most tracts generated are continuous from root into the spinal cord, cleaning is suggested. Find the level where the tracts enter the spinal cord, press Ctrl+S and draw a line across the spinal cord. This will ensure only tracts that pass that line will be kept.

![Asset 8](https://user-images.githubusercontent.com/43448105/191318832-e0b5d978-e7cb-4d73-b1a8-889e8b4931b9.png)

5. To save tracts, click on Tracts and save created tracts in the Output folder as a .txt file, named as previously described.
6. Repeat for each root level, side and location.

## Step 2: Simulate neural recruitment

This step is run from the code 'run_complex_model.py' and includes building a finite element model (FEM) based on the imaging data provided, simulates the electromagnetic potential evoked in that model during simulation, populates the model with neural trajectories, and simulates actual neural recruitment on those neurons. All of the neural recruitment data is then saved to an output file. To run this stage, open run_complex_model.py in a version of Sim4Life 7 or higher that includes A-delta neuron modeling capabilities, and in the "runProject" function change template and contact_n to be the name of the electrode file to load and the contact number to use, respectively. Change num_neuron to be the number of neurons per root and type with which to run the simulation. Additionally, pathnames in main() will need to be changed to local directories. 

After results are saved from the neuron simulation, run plot_analysis_scale.ipynb to re-scale these values to the correct ratio of neurons per root and view initial plots of recruitment values for a simulation. This step will also output a series of recruitment curve values in a text file that can be used in Step 3. 

## Step 3: Simulate bladder function

Step 3 includes a spiking neural network model that receives inputs generaged from upstream FEM model for experiment validation and exploration. 

The model needs [NEURON](https://www.neuron.yale.edu/neuron/) installed to perform normally. Please follow the *README.ipynb* notebook to compile the environment and preprocess the data before running a simulation.

We provide two options for running a simulation: 

1. use computed recruitment_threshold_percentage (only pudendal afferents are recruited and simulated) to validate bladder behavior under specific pudendal afferent recruitment

2. use recruitment curve data (pudendal afferent, pelvic afferent and SPN are recruited at different stimulation amplitude (uA)) and explore the bladder behavior at any customized stimulation amplitude by defining "stim_amp_uA". 

For option 2, the "splitting" and "mapping" steps in *recruitment data splitter.ipynb* is necessary.

All results are stored in ../results. There is also an analysis script Analysis on bladder traces and neuron firings.ipynb to provide some simple processing of data. The details of results are listed in *README.ipynb*.


