# Final Code With All Simulation (included anisotropic tensor conductivities)

#import components for the whole thing
from __future__ import absolute_import
from __future__ import print_function
import random
import sys, os
from os import listdir
import csv
sys.path.append('./')
import glob
import shutil
import numpy as np
#import pandas as pd
import math
import matplotlib.pyplot as plt
plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Arial']})

import s4l_v1.document as document
import XCoreModeling as xcore
import ImageModeling
import ViP
import MeshModeling
from XCoreModeling import LabelField
import s4l_v1.model as model
from s4l_v1.model import image
from s4l_v1.model import Vec3, Transform, Translation
import s4l_v1.analysis as analysis
import s4l_v1.simulation.emlf as emlf
from s4l_v1 import Unit
import s4l_v1.units as units
import XCoreModeling as xcore
import s4l_v1.simulation.neuron as neuron

from numpy.lib.npyio import save, zipfile_factory

from datetime import datetime

from helper_fn import *
from placeNeurons import *
from runNeuron import *
from plotAll import *

_CFILE = os.path.abspath(sys.argv[0] if __name__ == '__main__' else __file__ )
_CDIR = os.path.dirname(_CFILE)


# Load file and extract surface
def LoadImages(filename, dir_path=None):

    model_entities = []
    
    if dir_path is None:
        dir_path = os.path.abspath(os.path.join(_CDIR,'iSegCleaned'))
    
    print(dir_path)

    # set modelling unit to mm
    model.SetLengthUnits( units.MilliMeter )  # not reflected in GUI!
    print('Despite the display in the GUI, the model unit is', model.LengthUnits())

    
    filepath = os.path.join(dir_path, filename + '.prj')
    print('File path is: ' + filepath)
    assert os.path.exists(filepath), " Invalid path '%s'. Unable to import images." %filepath
    
    ents = xcore.Import(filepath) #ents is tuple
    
    label_fields = [e for e in ents if isinstance(e,ImageModeling.LabelField)]
    assert len(label_fields)==1, "expecting one LabelField"
    label_fields[0].Visible = False
    
    image_fields = [e for e in ents if isinstance(e,ImageModeling.Image)]
    image_fields[0].Visible = False
    image_fields[0].ApplyTransform(Translation(Vec3(12.8, -20, 12.1)))
    image_fields[0].ApplyTransform(Translation(Vec3(0, 0, 77)))
    
    #This crops 50 cleaned slices, from 100 to 150
    cropped_image = image.CropImage(label_fields[0], [0,383,0,511,10,700]) 
    cropped_image.Visible = False
    
    name = (filename + ' label Cropped')
    ents = model.AllEntities()
    seg = ents[name]
    
    #volume = MeshModeling.ExtractSurface(seg,  smooth=False,  num_iterations = 500, simplify = False)
    
    volume = image.ExtractSurface(cropped_image, smooth=False, num_iterations = 500, relaxation=1, simplify=False, avoid_self_intersections=True,  min_edge_length = 1.0)
    tissue_no = len(volume)
    print('there are ', tissue_no, ' tissues')
    componentlist = []
    
    # cil_start = [-5.579, 4.470036,50 ]
    # cil_end = [ -5.579, 4.470036,5]
    
    # cil_start = [-25.579, 0.470036,30 ]
    # cil_end = [ -25.579, 0.470036,-20]

    # saline = model.CreateSolidCylinder(Vec3(cil_start), Vec3(cil_end), 30)
    # saline.Name = 'saline'
    try:
        void = model.AllEntities()["void"]
        void.Delete()
    except:
        print('No void body detected in this file')
    
    tempbox = xcore.GetBoundingBox([model.AllEntities()["bone"]])
        
    
    # changed: box corner have a different meaning in a Cylinder and a Solid Box
    tempbox[0][2] -= 5
    tempbox[1][2] += 5
    
    # half of x and y size --> to get to the center    
    tempbox[0][0] += 12.05/2
    tempbox[0][1] += 7.55/2
    tempbox[1][0] -= 12.05/2
    tempbox[1][1] -= 7.55/2
    
    saline = model.CreateSolidCylinder(Vec3(tempbox[0]), Vec3(tempbox[1]), 20)    
    saline.Name = 'saline'
    extradura = saline.Clone()
    extradura.Name = 'extradura'
    
    model_entities = [x for x in model.AllEntities() if (x.Type == 'ENTITY_TRIANGLEMESH' or xcore.IsBody(x)) ]
     
    for ii in range(len(model_entities)):
        # print('Entity: ' + model_entities[ii].Name + ' is type ' + model_entities[ii].Type )
        # ignore if it's not a surface mesh
        if  (model_entities[ii].Type == 'ENTITY_TRIANGLEMESH' or xcore.IsBody(model_entities[ii])):
            if not ( model_entities[ii].Name == 'Lower DR' or model_entities[ii].Name == 'Lower VR' or model_entities[ii].Name == 'Lower WM' or model_entities[ii].Name == 'Upper DR' or model_entities[ii].Name == 'Upper VR' or model_entities[ii].Name == 'Upper WM'):
                componentlist.append(model_entities[ii].Name)
            
    print('Modelled tissues are: ', componentlist)
    
    return model_entities, componentlist



def buildElectrode(elec_dim, array_dim, elec_start, array_start, elec_spacing):
    '''Take top left corner of array, bottom edge, and extend array from there
    '''
    from s4l_v1.model import Vec3, Rotation, Translation
    import math

    elec_end = []
    array_end = []
    for i in range(len(elec_dim)):
        elec_end.append(elec_start[i] + elec_dim[i])
        array_end.append(array_start[i] + array_dim[i])


    # print('elec start:', elec_start)
    # print('elec end:', elec_end)
    #start = elec_start + elec_dim
    
    contact = model.CreateSolidBlock(Vec3(elec_start), Vec3(elec_end))
    contact.Name = 'contact'
    
    electrode = xcore.CreateGroup('electrode')

    from XCoreModeling import EntityGroup
    # EntityGroup.Add(electrode,silicone)
    EntityGroup.Add(electrode, contact)

    
    silicone = model.CreateSolidBlock(Vec3(array_start), Vec3(array_end))
    silicone.Name = 'silicone'
	
    from XCoreModeling import EntityGroup
    EntityGroup.Add(electrode,silicone)
    EntityGroup.Add(electrode,contact)
	
    xcore.BendEntity(silicone, Vec3([0,30,2]),Vec3([0,0,1]),Vec3([0,-1,0]),5 ,1, width=5, center=True)
    xcore.BendEntity(contact, Vec3([0,31,-2.5]),Vec3([0,0,1]),Vec3([0,-1,0]),5 ,1, width=5, center=True)	
	
    
    return [contact, silicone]
    
def createStructuredEMSim(model_components):


    '''Makes a grid and EM sim - input model components object list (ie includes names, conductivities)'''
    sim = emlf.ElectroQsOhmicSimulation()
    sim.Name = 'EM Simulation'
    entities = model.AllEntities() #should pull all of the electrode and tissues

    quasi_static_setup_settings = [x for x in sim.AllSettings if isinstance(x, emlf.QuasiStaticSetupSettings) and x.Name == "Setup"][0]
    quasi_static_setup_settings.Frequency = 0, units.Hz

    #Make grid mask the size of the saline bounds + a little
    tempbox = xcore.GetBoundingBox([entities['saline']])
    for i in range(3):
        tempbox[0][i] -= 5
        tempbox[1][i] += 5
    gridMask = model.CreateSolidBlock(tempbox[0], tempbox[1], True)    
    gridMask.Name = 'Grid Mask'

    #add a new materialsettings for each
    #Here we exclude roots and wm, to which we assign the conductivity tensor previously calculater
    for i in range(len(model_components)):
        material_settings = emlf.MaterialSettings()
        
        # Do not assign material setting to the contact
        if not (model_components[i].name == 'contact' or model_components[i].name == 'saline' or model_components[i].name == 'whitematter' or model_components[i].name == 'dorsalroots' or model_components[i].name == 'ventralroots'):
            material_settings.name = "Material settings " + model_components[i].name
            if type(model_components[i].dielectric) is list:
                material_settings.ElectricProps.ConductivityAnisotropic = True
                material_settings.ElectricProps.ConductivityDiagonalElements = np.array(model_components[i].dielectric)#, Unit('S/m')
            elif type(model_components[i].dielectric) is str:
                material_settings.MaterialType = material_settings.MaterialType.enum.PEC
            else:
                material_settings.ElectricProps.Conductivity = model_components[i].dielectric#, Unit('S/m')

            sim.Add(material_settings, [entities[model_components[i].name]])
        # sim.Remove(material_settings, [entities['contact']])
    

    entity_wm = entities['whitematter']
    roots_d = entities['dorsalroots']
    roots_v = entities['ventralroots']
    
    # White matter
    material_settings = emlf.MaterialSettings()
    components = entity_wm
    material_settings.Name = "WM"
    conductivity_algorithm = document.AllAlgorithms['Anisotropic Conductivity']
    wm_conductivity = conductivity_algorithm.Outputs['Anisotropic Conductivity From E Field']
    material_settings.SetInhomogeneousConductivity(wm_conductivity)
    sim.Add(material_settings, components)

    # Rootlets
    material_settings = emlf.MaterialSettings()
    components = [roots_d] + [roots_v]
    material_settings.Name = "Rootlets"
    conductivity_algorithm = document.AllAlgorithms['Anisotropic Conductivity']
    wm_conductivity = conductivity_algorithm.Outputs['Anisotropic Conductivity From E Field']
    material_settings.SetInhomogeneousConductivity(wm_conductivity)
    sim.Add(material_settings, components)
    
    # Removing AutomaticGridSettings Automatic
    automatic_grid_settings = [x for x in sim.AllSettings if isinstance(x, emlf.AutomaticGridSettings) and x.Name == "Automatic"][0]
    sim.RemoveSettings(automatic_grid_settings)

    #Add a new Manual Grid Settings for each tissue layer
    for i in range(len(model_components)):
        #in general, think of the resolution as a pseudo minimum step
        maxstep = model_components[i].localmesh
        minstep = 2
        manual_grid_settings = sim.AddManualGridSettings([entities[model_components[i].name]])
        manual_grid_settings.MaxStep = np.array([maxstep, maxstep, maxstep]), units.MilliMeters
        manual_grid_settings.Resolution = np.array([minstep, minstep, minstep]), units.MilliMeters
    
    # Removing AutomaticVoxelerSettings Automatic Voxeler Settings
    automatic_voxeler_settings = [x for x in sim.AllSettings if isinstance(x, emlf.AutomaticVoxelerSettings) and x.Name == "Automatic Voxeler Settings"][0]
    sim.RemoveSettings(automatic_voxeler_settings)

    # Adding a new ManualVoxelerSettings for each
    for i in range(len(model_components)):
        manual_voxeler_settings = emlf.ManualVoxelerSettings()
        manual_voxeler_settings.Name = "Manual Voxeler Settings " + model_components[i].name
        manual_voxeler_settings.Priority = model_components[i].priority
        sim.Add(manual_voxeler_settings, [entities[model_components[i].name]])

    
    #TODO Somewhere in here define the stim amplitude!!
    #TODO try adding stim, boundaries, thin layers manually and then export to python!!
    #TODO no boundary settings here yet
    # boundary = sim.AllSettings['Boundary Settings']
    # sim.RemoveSettings(boundary)

    # Sets current density (A/m^2) * electrode SA (mm^2) = current amplitude (uA) 
    # Note 1 uA/mm^2 = 1 A/m^2 due to unit cancellation
    # current_dens = round(current_uA/elec_sa_mm, 2)

    # Adding a new BoundarySettings
    boundary_settings = emlf.BoundarySettings()
    boundary_settings.Name = "Boundary Settings 1"
    boundary_settings.BoundaryType = boundary_settings.BoundaryType.enum.Dirichlet
    boundary_settings.DirichletValue = +10 #current_dens, Unit("A/m^2")
    sim.Add(boundary_settings, entities['contact'])

    #UNCOMMENT THIS IF YOU WANT TO HAVE A BOUNDAY CONDITION ON SALINE
    #Adding a new BoundarySettings
    boundary_settings = emlf.BoundarySettings()
    boundary_settings.Name = "Boundary Settings 2"
    boundary_settings.BoundaryType = boundary_settings.BoundaryType.enum.Dirichlet
    boundary_settings.DirichletValue = 0.0
    sim.Add(boundary_settings, entities["saline"])

    #HOW it works in unstructured model:
    # for bond in model_components.boundarieslist:
        # boundary_settings = emlf.BoundarySettings()
        # boundary_settings.Name = bond.entity.Name
        # if bond.type == 'Flux':
            # boundary_settings.BoundaryType = boundary_settings.BoundaryType.enum.Flux
            # boundary_settings.FluxValue = bond.value
        # elif bond.type == 'Dirichlet':
            # print('Add dirichlet boundary')
            # boundary_settings.BoundaryType = boundary_settings.BoundaryType.enum.Dirichlet
            # boundary_settings.DirichletValue = bond.value
        # sim.Add(boundary_settings, [bond.entity])
    #TODO add dura (thin layers)

    # Editing FieldSensorSettings
    field_sensor_settings = [x for x in sim.AllSettings if isinstance(
        x, emlf.FieldSensorSettings) and x.Name == "Field Sensor Settings"][0]
    field_sensor_settings.RecordHField = True
    field_sensor_settings.RecordVectorPotentialField = True

    # Adding SolverSettings
    solver_settings = sim.SolverSettings
    solver_settings.PredefinedTolerances = (solver_settings.PredefinedTolerances.enum.UserDefined)
    sim.SolverSettings.RelativeSolverTolerance = 1e-18
    sim.UpdateAllMaterials()
    sim.UpdateGrid()
    document.AllSimulations.Add(sim)

    return sim
    
def ScaleAndAnalyze(simEM):
    # Adding a new ModelToGridFilter
    simulation_extractor = simEM.Results() 
    
    inputs = []
    model_to_grid_filter = analysis.core.ModelToGridFilter(inputs=inputs)
    model_to_grid_filter.Name = "flux sphere"
    model_to_grid_filter.Entity = model.AllEntities()["flux sphere"]
    model_to_grid_filter.MaximumEdgeLength = 0.025980762481689453, units.Meters
    model_to_grid_filter.UpdateAttributes()
    document.AllAlgorithms.Add(model_to_grid_filter)

    # Adding a new EmSensorExtractor
    em_sensor_extractor = simulation_extractor["Overall Field"]
    em_sensor_extractor.FrequencySettings.ExtractedFrequency = u"All"
    document.AllAlgorithms.Add(em_sensor_extractor)

    # Adding a new FieldFluxEvaluator
    inputs = [em_sensor_extractor.Outputs["J(x,y,z,f0)"], model_to_grid_filter.Outputs["Surface"]]
    field_flux_evaluator = analysis.core.FieldFluxEvaluator(inputs=inputs)
    field_flux_evaluator.UpdateAttributes()
    document.AllAlgorithms.Add(field_flux_evaluator)
    field_flux_evaluator.Update()
    flux = np.real(field_flux_evaluator.Outputs[1].Data.GetComponent(0))
    print('flux is: ',flux)

    scaling_factor2 = 1/(flux*1000)
    scaling_factor = float(scaling_factor2)
    print(scaling_factor)



    # Creating the analysis pipeline
    # Adding a new SimulationExtractor
    simulation = document.AllSimulations["EM Simulation"]
    simulation_extractor = simulation.Results()

    # Adding a new EmSensorExtractor
    em_sensor_extractor = simulation_extractor["Overall Field"]
    em_sensor_extractor.FrequencySettings.ExtractedFrequency = u"All"
    document.AllAlgorithms.Add(em_sensor_extractor)

    #CURRENT FLUX
    # Adding a new UserDefinedFieldNormalizer for flux
    inputs = [em_sensor_extractor.Outputs["J(x,y,z,f0)"]]
    user_defined_field_normalizer = analysis.field.UserDefinedFieldNormalizer(inputs=inputs)
    user_defined_field_normalizer.Target.Value = scaling_factor
    user_defined_field_normalizer.UpdateAttributes()
    document.AllAlgorithms.Add(user_defined_field_normalizer)

    # Adding a new VectorFieldViewer
    inputs = [user_defined_field_normalizer.Outputs["J(x,y,z,f0)"]]
    vector_field_viewer = analysis.viewers.VectorFieldViewer(inputs=inputs)
    vector_field_viewer.Data.Phase = u"0°"
    vector_field_viewer.Vector.Plane.PlaneCenter = Vec3(-0.027745, -0.0027455, -0.034)#, units.Meters
    vector_field_viewer.Vector.Plane.Resolution = 500
    vector_field_viewer.Vector.ArrowSize = 0.28866589069366455
    vector_field_viewer.UpdateAttributes()
    document.AllAlgorithms.Add(vector_field_viewer)
    
    #EM POTENTIAL
    # Adding a new UserDefinedFieldNormalizer
    inputs = [em_sensor_extractor.Outputs["EM Potential(x,y,z,f0)"]]
    user_defined_field_normalizer = analysis.field.UserDefinedFieldNormalizer(inputs=inputs)
    user_defined_field_normalizer.Target.Value = scaling_factor
    user_defined_field_normalizer.UpdateAttributes()
    document.AllAlgorithms.Add(user_defined_field_normalizer)
    
    # Adding a new SliceFieldViewer
    inputs = [user_defined_field_normalizer.Outputs["EM Potential(x,y,z,f0)"]]
    slice_field_viewer = analysis.viewers.SliceFieldViewer(inputs=inputs)
    slice_field_viewer.Data.Mode = slice_field_viewer.Data.Mode.enum.QuantityRealPart
    slice_field_viewer.Slice.Plane = slice_field_viewer.Slice.Plane.enum.YZ
    slice_field_viewer.Slice.Index = 356
    slice_field_viewer.Visualization.Smooth = True
    slice_field_viewer.UpdateAttributes()
    document.AllAlgorithms.Add(slice_field_viewer)
    
    # Adding a new EmSensorExtractor
    em_sensor_extractor = simulation_extractor["Overall Field"]
    em_sensor_extractor.FrequencySettings.ExtractedFrequency = u"All"
    document.AllAlgorithms.Add(em_sensor_extractor)

    # Adding a new UserDefinedFieldNormalizer
    inputs = [em_sensor_extractor.Outputs["EM Potential(x,y,z,f0)"]]
    user_defined_field_normalizer = analysis.field.UserDefinedFieldNormalizer(inputs=inputs)
    user_defined_field_normalizer.Target.Value = scaling_factor
    user_defined_field_normalizer.UpdateAttributes()
    document.AllAlgorithms.Add(user_defined_field_normalizer)

    # Adding a new SliceFieldViewer
    inputs = [user_defined_field_normalizer.Outputs["EM Potential(x,y,z,f0)"]]
    slice_field_viewer = analysis.viewers.SliceFieldViewer(inputs=inputs)
    slice_field_viewer.Data.Mode = slice_field_viewer.Data.Mode.enum.QuantityRealPart
    slice_field_viewer.Slice.Plane = slice_field_viewer.Slice.Plane.enum.YZ
    slice_field_viewer.Slice.Index = 224
    slice_field_viewer.Visualization.Smooth = True
    slice_field_viewer.UpdateAttributes()
    document.AllAlgorithms.Add(slice_field_viewer)
    
    # Adding a new UserDefinedFieldNormalizer
    inputs = [em_sensor_extractor.Outputs["EM Potential(x,y,z,f0)"]]
    user_defined_field_normalizer = analysis.field.UserDefinedFieldNormalizer(inputs=inputs)
    user_defined_field_normalizer.Target.Value = scaling_factor
    user_defined_field_normalizer.UpdateAttributes()
    document.AllAlgorithms.Add(user_defined_field_normalizer)

    
def exportEM(simEM, loc, savepath):
    '''simEM is an Unstructured LF simulation, loc is the z location of the field cross section, 
    savepath is an empty string or if saving is the location to save the text file'''
    #add a SimulationExtractor
    simulation_extractor = simEM.Results()
    #add a sensor extractor
    em_sensor_extractor = simulation_extractor["Overall Field"]
    em_sensor_extractor.FrequencySettings.ExtractedFrequency = u"All"
    document.AllAlgorithms.Add(em_sensor_extractor)
    # add a field gradient evaluator
    # inputs = [em_sensor_extractor.Outputs["EM Potential(x,y,z,f0)"]]
    # field_gradient_evaluator = analysis.core.FieldGradientEvaluator(inputs=inputs)
    # field_gradient_evaluator.UpdateAttributes()
    # document.AllAlgorithms.Add(field_gradient_evaluator)

    # Adding a new SliceFieldViewer
    inputs = [em_sensor_extractor.Outputs["EM E(x,y,z,f0)"]]
    slice_field_viewerE = analysis.viewers.SliceFieldViewer(inputs=inputs)
    slice_field_viewerE.Data.Mode = slice_field_viewerE.Data.Mode.enum.QuantityAbsolute
    slice_field_viewerE.Data.Component = slice_field_viewerE.Data.Component.enum.Component0
    slice_field_viewerE.Slice.SliceOrigin = np.array([0.0, -0.001294769172091037, 0.010628291638568044])
    slice_field_viewerE.Slice.Normal = np.array([0.0, 1.0, 0.0])
    slice_field_viewerE.Visualization.Transparency = 0.0
    slice_field_viewerE.Visualization.Smooth = True
    slice_field_viewerE.UpdateAttributes()
    document.AllAlgorithms.Add(slice_field_viewerE)

    # Adding a new SliceFieldViewer
    inputs = [em_sensor_extractor.Outputs["EM Potential(x,y,z,f0)"]]
    slice_field_viewerV = analysis.viewers.SliceFieldViewer(inputs=inputs)
    slice_field_viewerV.Data.Mode = slice_field_viewerV.Data.Mode.enum.QuantityAbsolute
    slice_field_viewerV.Slice.SliceOrigin = np.array([0.0, -0.0017000000225380063, 0.04207948222756386])
    slice_field_viewerV.Slice.Normal = np.array([0.0, 1.0, 0.0])
    slice_field_viewerV.Visualization.Transparency = 0.0
    slice_field_viewerV.UpdateAttributes()
    document.AllAlgorithms.Add(slice_field_viewerV)

    # #add a vector field viewer
    # inputs = [field_gradient_evaluator.Outputs["grad(EM Potential(x,y,z,f0))"]]
    # vector_field_viewer = analysis.viewers.VectorFieldViewer(inputs=inputs)
    # vector_field_viewer.Data.Phase = u"0°"
    # vector_field_viewer.Vector.Plane.Resolution = 60
    # vector_field_viewer.Vector.ArrowSize = 1.2
    # vector_field_viewer.Vector.Plane.PlaneCenter = np.array([0.0, 0.0, loc])
    # vector_field_viewer.UpdateAttributes()
    # document.AllAlgorithms.Add(vector_field_viewer)
    # vector_field_viewer.Vector.Plane.PlaneCenter = np.array([0.0, 0.0, loc])
    # vector_field_viewer.UpdateAttributes()
    # document.AllAlgorithms.Add(vector_field_viewer)

    #MUST DO THIS STEP FIRST: 
    # field_gradient_evaluator.Update()
    slice_field_viewerV.Update()
    slice_field_viewerE.Update()

    if savepath: 
        file1 = open(savepath, 'w+')
        file1.write('X,Y,Z,V,E\n')

        #get all the points in the sensor and format the same as Comsol export files
        for i in range(em_sensor_extractor.Outputs[4].Data.Grid.NumberOfPoints):
            temppoint = em_sensor_extractor.Outputs[4].Data.Grid.GetPoint(i)
            appendstr = str(temppoint[0]) + ',' + str(temppoint[1]) + ',' + str(temppoint[2]) + ',' + str(em_sensor_extractor.Outputs[4].Data.FieldValue(0, i, 0).real) + ',' + str(em_sensor_extractor.Outputs[0].Data.FieldValue(0, i, 0).real) + '\n'
            #print(appendstr)
            file1.write(appendstr)
            if em_sensor_extractor.Outputs[4].Data.FieldValue(0, i, 0).imag != 0:
                print('Non zero imaginary component at idx: ' + str(i))

        file1.close()
    #save to text file: X, Y, Z, Potential
    

    
def setParams(componentlist):
    #component list is a list of strings representing names of the tissues to be imported
    #arraytype is a string; allowed values are: 'mux', 'microleads3x8', [add more later]
    # arraytype = arraytype.replace(' ', '').lower()
    
    #TODO: correct these values
    conduct_dict = {'graymatter':.23, 'whitematter':[.083, .083, .6], 'csf':1.7, 'epiduralfat':.04, 'dorsalroots':[.083, .083, .6], 'ventralroots':[.083, .083, .6], 'bone':0.02,  'saline':0.2, 'extradura':0.2, 'dura':0.03}
    meshsize_dict = {'graymatter':1, 'whitematter':0.3,  'csf':1, 'epiduralfat':0.7, 'dorsalroots':0.3, 'ventralroots':0.3,'bone':4, 'saline':5, 'extradura':5, 'dura':.05}
    priority_dict = {'graymatter':8, 'whitematter':7,  'csf':5, 'epiduralfat':3, 'dorsalroots':6, 'ventralroots':6,'bone':2, 'saline':0, 'extradura':1, 'dura':4}
    #Setting priorities: electrode silicone + contact must be higher priority than everything except white and gray matter
    #Also, in theory the two white matter parts are the same but the mesher calls that a "non regularized unite" and fails

    model_components = []
    for comp in componentlist:
        model_components.append(ModelComponent(comp, conduct_dict[comp], meshsize_dict[comp], priority_dict[comp]))

    #print('TODO: make sure white matter array is oriented correctly')
    
    #define information for different array types
    #critical note on array and electrode dimensions: in the orientation of the spinal cord here, this is as follows - 
    #x: ventral->dorsal (thickness) y: right->left (width) z: rostral->caudal (length)
    # if 'mux' in arraytype:
    
    # NOTA ARIANNA 33.1 WAS THE VALUE Z OF ARRAY DIM
    elec_dim = [.05, 0.291, 1.0]
        # elec_dim = [.45,.05,  1.35] #x, y, z NOTE x (thickness) is an estimate right now
        # array_dim = [ 4.53, .34, 32.41 ] #x, y, z TODO array is actually split halfway .34 and .56
        # elec_offset = [ .51, 0, 1] #add these values to get to top right electrode
        # elec_spacing = [ 0.57, 0, 3.19] #offset for additional electrodes from corner
    # elif '3x8' in arraytype:
        # TODO fix the array start stuff (come up with values for each level?)
        # elec_dim = [.05, 0.291, 1.0] #NOTE: x is an estimate
        # array_dim = [.34, 4.5, 10.014] #NOTE: x is an estimate
        # elec_offset = [0, 0.3, 2.73] #add these values to get to top right electrode
        # elec_spacing = [0, 0.225, 0.775] #offset for additional electrodes from corner
        
        # elec_dim = [0.291, 0.05, 1.02] #NOTE: x is an estimate
        # array_dim = [4.5, .34, 10.014] #NOTE: x is an estimate
        # elec_offset = [0.3, 0, 6.284] #add these values to get to top right electrode
        # elec_spacing = [ 0.225,0, -0.775] #offset for additional electrodes from corner
        
        # elec_dim = [0.291, .05, 1.0] #NOTE: x is an estimate
        # array_dim = [ 4.5, .34, 10.014] #NOTE: x is an estimate
        # elec_offset = [0, 0.3, 2.73] #add these values to get to top right electrode
        # elec_spacing = [0, 0.225, 0.775] #offset for additional electrodes from corner
    
    # elec_params = {'elec_dim':elec_dim, 'array_dim':array_dim, 'elec_offset':elec_offset, 'elec_spacing':elec_spacing}
    elec_params = {'elec_dim':elec_dim}
    
    return model_components, elec_params
    
def CreateLFSimulation(project_path):
    #Here we load bondiaries already created. Option to do them manually when moving to OSPARC
    xcore.Import('P:/users/ard171/scs_modeling/cat_model_3/Boundaries/ComplexBoundaries_V4.smash')

    """
    Set up all the settings of the EM-LF simn 
    """
    sim = emlf.ElectroQsOhmicSimulation()
    sim.Name = "Subject_Anisotropy " 
    print("Creating simulation for %s" % sim.Name)
    

    #Make grid mask the size of the saline bounds + a little
    entities = model.AllEntities()
    tempbox = xcore.GetBoundingBox([entities['saline']])
    for i in range(3):
        tempbox[0][i] -= 5
        tempbox[1][i] += 5
    gridMask = model.CreateSolidBlock(tempbox[0], tempbox[1], True)
    gridMask.Name = 'Grid Mask'

    ents = model.AllEntities()
    
    # Mapping the components and entities
    component_plane_x = sim.AllComponents["Plane X+"]
    component_plane_x_1 = sim.AllComponents["Plane X-"]
    component_plane_y = sim.AllComponents["Plane Y+"]
    component_plane_y_1 = sim.AllComponents["Plane Y-"]
    component_plane_z = sim.AllComponents["Plane Z+"]
    component_plane_z_1 = sim.AllComponents["Plane Z-"]
    
    # Get all Entities
    entity_wm = ents['whitematter']
    roots_d = ents['dorsalroots']
    roots_v = ents['ventralroots']
    entity_lower_dr = ents['Lower DR'] 
    entity_upper_dr = ents['Upper DR']
    entity_lower_vr = ents['Lower VR'] 
    entity_upper_vr = ents['Upper VR']
    entity_lower_wm = ents['Lower WM'] 
    entity_upper_wm = ents['Upper WM']
    entity_saline = ents["saline"]
    #entity_bone = ents["Bone"]
    #entity_fat = ents["Fat"]
    entity_csf = ents["csf"]
    entity_gm = ents["graymatter"]
    # entity_ipg_face = ents["IPG"]
    
    Lead_comp = ents['contact']
    Paddle = ents['silicone']
    
    ### Setup
    sim.SetupSettings.Frequency = 1000, units.Hz
    
    ### Materials
    # White matter
    material_settings = emlf.MaterialSettings()
    components = [entity_wm] + [roots_d] + [roots_v]
    material_settings.Name = "WM & Roots"
    material_settings.ElectricProps.Conductivity = 1, Unit("S/m")
    sim.Add(material_settings, components)
    
    ### Boundary Conditions
    # Adding BoundarySettings for 1V at active site
    boundary_settings = emlf.BoundarySettings()
    components = [entity_lower_dr] + [entity_lower_vr] + [entity_lower_wm]
    boundary_settings.Name = "Lower Boundary"
    boundary_settings.DirichletValue = 1.0, units.Volts
    sim.Add(boundary_settings, components)

    # Adding BoundarySettings for 0V at skin-air interface
    boundary_settings = emlf.BoundarySettings()
    components = [entity_upper_dr] + [entity_upper_vr] + [entity_upper_wm]
    boundary_settings.Name = "Upper Boundary"
    boundary_settings.DirichletValue = 0.0, units.Volts
    sim.Add(boundary_settings, components)
    
    # Removing AutomaticGridSettings Automatic
    automatic_grid_settings = [x for x in sim.AllSettings if isinstance(x, emlf.AutomaticGridSettings) and x.Name == "Automatic"][0]
    sim.RemoveSettings(automatic_grid_settings)

    #Add a new Manual Grid Settings for each tissue layer
    # for i in range(len(model_components)):
    # in general, think of the resolution as a pseudo minimum step
    # maxstep = model_components[i].localmesh
    # minstep = 2
    # manual_grid_settings = sim.AddManualGridSettings([entities[model_components[i].name]])
    # manual_grid_settings.MaxStep = numpy.array([maxstep, maxstep, maxstep]), units.MilliMeters
    # manual_grid_settings.Resolution = numpy.array([minstep, minstep, minstep]), units.MilliMeters

    # Removing AutomaticVoxelerSettings Automatic Voxeler Settings
    automatic_voxeler_settings = [x for x in sim.AllSettings if isinstance(x, emlf.AutomaticVoxelerSettings) and x.Name == "Automatic Voxeler Settings"][0]
    sim.RemoveSettings(automatic_voxeler_settings)

    ### Grid
    #Editing GlobalGridSettings "Grid (Empty)"
    # global_grid_settings = sim.GlobalGridSettings
    # global_grid_settings.PaddingMode = global_grid_settings.PaddingMode.enum.Manual
    # global_grid_settings.BottomPadding = np.array([1.0, 1.0, 1.0]), units.MilliMeters
    # global_grid_settings.TopPadding = np.array([1.0, 1.0, 1.0]), units.MilliMeters
    # global_grid_settings.ManualDiscretization = True
    # global_grid_settings.MaxStep = (1,)*3 # m.u
    # global_grid_settings.Resolution = (2, )*3 #m.u.
    
    # Adding a new AutomaticGridSettings
    manual_grid_settings = sim.AddManualGridSettings([entity_gm, entity_csf])
    manual_grid_settings.Name = "CSF_GM"
    manual_grid_settings.MaxStep = np.array([1, 1, 1]), units.MilliMeters
    manual_grid_settings.Resolution = np.array([2, 2, 2]), units.MilliMeters

    # Adding a new AutomaticGridSettings
    manual_grid_settings = sim.AddManualGridSettings([entity_lower_dr,entity_lower_vr,entity_lower_wm, entity_upper_dr,entity_upper_vr,entity_upper_wm])
    manual_grid_settings.Name = "Boundaries"
    manual_grid_settings.MaxStep = np.array([1, 1, 1]), units.MilliMeters
    manual_grid_settings.Resolution = np.array([2, 2, 2]), units.MilliMeters

    # Manual fine grid for WM
    manual_grid_settings = sim.AddManualGridSettings([entity_wm] + [roots_d] + [roots_v])
    manual_grid_settings.Name = "WM"
    manual_grid_settings.MaxStep = np.array([0.3, 0.3, 0.3]), units.MilliMeters
    manual_grid_settings.Resolution = np.array([2, 2, 2]), units.MilliMeters

    # Manual fine grid for electrodes
    manual_grid_settings = sim.AddManualGridSettings(Lead_comp)
    manual_grid_settings.Name = "Electrodes"
    manual_grid_settings.MaxStep = np.array([1, 1, 1]), units.MilliMeters

    # Course grid for saline
    components = [entity_saline]
    manual_grid_settings = sim.AddManualGridSettings(components)
    manual_grid_settings.Name = "Saline_Grid"
    manual_grid_settings.MaxStep = np.array([5.0, 5.0, 5.0]), units.MilliMeters
    manual_grid_settings.Resolution = np.array([2, 2, 2]), units.MilliMeters
  
    # Sensors
    # Record everything
    sensor_settings = next(x for x in sim.AllSettings if x.Name == "Field Sensor Settings")

    ### Voxels
    # Set Voxels for WM & Roots
    manual_voxeler_settings = emlf.ManualVoxelerSettings()
    components = [entity_wm] + [roots_d] + [roots_v]
    manual_voxeler_settings.Priority = 0
    sim.Add(manual_voxeler_settings, components)

    # Set Voxels for Boundaries
    manual_voxeler_settings = emlf.ManualVoxelerSettings()
    components = [entity_lower_dr,entity_lower_vr,entity_lower_wm, entity_upper_dr,entity_upper_vr,entity_upper_wm]
    manual_voxeler_settings.Priority = 1
    sim.Add(manual_voxeler_settings, components)

    solver_settings_1 = sim.SolverSettings
    solver_settings_1.PredefinedTolerances = solver_settings_1.PredefinedTolerances.enum.High
    solver_settings_1.AdditionalSolverOptions = u"-ksp_type gmres"

    
    sim.UpdateAllMaterials()
    sim.UpdateGrid()
    sim.CreateVoxels(project_path)
    document.AllSimulations.Add(sim)
    
    sim.WriteInputFile()
    
    
    return sim
    

def CalculateAnisotropicConductivity(simname, doc):
    # Transversal and longitudinal sigmas
    sigma1 = 0.6
    sigma2 = 0.083

    # here conductivity has 6 component (anisotropic tensor)
    sim = doc.AllSimulations[simname]
    results = sim.Results()
    p = results["Overall Field"].Outputs["EM Potential(x,y,z,f0)"]
    p.Update()
    e = results["Overall Field"].Outputs["EM E(x,y,z,f0)"]
    e.Update()

    e_field = e.Data.Field(0)
    nc = e.Data.Grid.NumberOfCells

    # Calculate sigma for white matter on cell centers
    print("Calculating anisotropic sigma values...")
    sigma_wm = np.zeros([nc,6]) # 6 unique values
    iu3 = (np.array([0, 1, 2, 0, 1, 0], dtype=np.int64), np.array([0, 1, 2, 1, 2, 2], dtype=np.int64))
    for idx, val in enumerate(e_field):
        tmp = np.zeros([3,3], dtype=np.complex)
        tmp[0] = val
        if (np.isnan(val).any()):
            #P = np.full((3, 3), np.nan) #, dtype=np.complex) <-- nans cause the preconditioner to hang
            P = np.full((3, 3), 0, dtype=np.complex)
        else:
            if (np.inner(val, val) != 0):
                P = np.dot(tmp.T, tmp) / np.inner(val,val)
            else:
                P = np.full((3, 3), 0, dtype=np.complex)
        sigma_tensor = sigma1*P + sigma2*( np.identity(P.shape[0]) - P ) 
        sigma_wm[idx] = sigma_tensor[iu3].real
    
    # here conductivity has 6 component (anisotropic tensor)
    results = sim.Results()
    p = results["Overall Field"].Outputs["EM Potential(x,y,z,f0)"]
    p.Update()
    e = results["Overall Field"].Outputs["EM E(x,y,z,f0)"]
    e.Update()
    
    tensor_conductivity = analysis.core.DoubleFieldData()
    tensor_conductivity.NumberOfSnapshots = 1
    tensor_conductivity.NumberOfComponents = 6
    tensor_conductivity.Grid = e.Data.Grid.Clone()
    tensor_conductivity.ValueLocation = analysis.core.eValueLocation.kCellCenter
    tensor_conductivity.SetField(0, sigma_wm)
    tensor_conductivity.Quantity.Name = 'Anisotropic Conductivity From E Field'
    tensor_conductivity.Quantity.Unit = analysis.core.Unit('S/m')

    posName = simname.split(" ")[-1]
    producer_t = analysis.core.TrivialProducer()
    producer_t.SetDataObject(tensor_conductivity)
    producer_t.Name = 'Anisotropic Conductivity'
    producer_t.Update()
    document.AllAlgorithms.Add(producer_t)
    
    # exporter = s4l.analysis.exporters.DataCacheExporter()
    # exporter.Inputs[0].Connect(producer_t.Outputs[0])
    # exporter.FileName = os.path.join(output_dir, r'{}.cache'.format(simname))
    # exporter.Update()    
    
def InterpolateAnisotropicConductivity(simname, doc):
    # Adding a new SimulationExtractor
    sim = doc.AllSimulations[simname]
    simulation_extractor = sim.Results()

    # Adding a new EmSensorExtractor
    em_sensor_extractor = simulation_extractor["Overall Field"]
    em_sensor_extractor.FrequencySettings.ExtractedFrequency = u"All"
    doc.AllAlgorithms.Add(em_sensor_extractor)

    # Get producer for anisotropic conductivity algorithm
    producer_t = doc.AllAlgorithms['Anisotropic Conductivity']
    print(producer_t)

    # Adding a new FieldInterpolationFilter
    inputs = [producer_t.Outputs["Anisotropic Conductivity From E Field"], em_sensor_extractor.Outputs["EM E(x,y,z,f0)"]]
    field_interpolation_filter = analysis.core.FieldInterpolationFilter(inputs=inputs)
    field_interpolation_filter.UpdateAttributes()
    field_interpolation_filter.Name = 'Anisotropic Conductivity '
    doc.AllAlgorithms.Add(field_interpolation_filter)


def runProject(project_path, model_path):
    
    #make document
    document.New()
    #document.Open('C:/Users/ard171/Arianna/FEM_Model_Arianna/Taylor_Running_Conductivities.smash')
    #document.Open('P:/users/ard171/scs_modeling/cat_model_1/Boundaries/ComplexBoundaries.smash')
    document.SaveAs(project_path)

    global stim_params
    global runtime
    runtime = datetime.now()
    data_dir = os.path.abspath(os.path.join(_CDIR, '..', 'saved_data_finalmodel'))
    #STIM PARAMS
    run_type = 'full_run' #'full_run', place_only
    num_neuron = 25 #this is PER NERVE+AFFERENT/EFF GROUP and it must match the number of neurons in a given test set if loading from saved file
    catname = 'cat1'
    loadcat = True
    savecat = False 
    template = 'L7'  # Can be either L6,L7,S1
    contact_n = '14'  # Can be from 1 to 24; 8 is left side ("ipsilateral")

    print('Run ' + run_type + ' with # neurons: ' + str(num_neuron*20))
    print('Neuron set from ' + catname + ' stim at ' + template + ' channel ' + contact_n)
    stim_params = {'current_uA':1000, 'stim_start_ms':0.1, 'stim_pw_ms':0.2}
    if run_type == 'model_only':
        run_parts = {'importModel':True, 'makeElectrode':False, 'WM_Conductivities':False, 'placeNeurons':False, 'screenNeurons':False, 'makeStructuredMesh':False, 'runEM':False, 'ScaleAndAnalyze':False, 'exportEM':False, 'runNeuron':False, 'plotAll':False}
    elif run_type == 'place_only':
        run_parts = {'importModel':True, 'makeElectrode':True, 'WM_Conductivities':False, 'placeNeurons':True, 'screenNeurons':False, 'makeStructuredMesh':False, 'runEM':False, 'ScaleAndAnalyze':False, 'exportEM':False, 'runNeuron':False, 'plotAll':False}
        #also update so saving figs doesn't happen on these
    elif run_type == 'resave_neurons':
        #screen the neurons to make sure they  for the correct info and save them. 
        run_parts = {'importModel': True, 'makeElectrode': True, 'WM_Conductivities': False, 'placeNeurons': False, 'screenNeurons':True,
                     'makeStructuredMesh': False, 'runEM': False, 'ScaleAndAnalyze': False, 'exportEM': False, 'runNeuron': False, 'plotAll': False}
    elif run_type == 'mesh':
        run_parts = {'importModel':True, 'makeElectrode':True, 'WM_Conductivities':True, 'placeNeurons':True, 'screenNeurons':False, 'makeStructuredMesh':True, 'runEM':True, 'ScaleAndAnalyze':True, 'exportEM':False, 'runNeuron':True, 'plotAll':False}
    elif run_type == 'full_run':
        run_parts = {'importModel':True, 'makeElectrode':True, 'WM_Conductivities':True, 'placeNeurons':True, 'screenNeurons':False, 'makeStructuredMesh':True, 'runEM':True, 'ScaleAndAnalyze':True, 'exportEM':False, 'runNeuron':True, 'plotAll':True}

    # run_parts = {'importModel': True, 'makeElectrode': True, 'WM_Conductivities': True, 'placeNeurons': False,
    #              'makeStructuredMesh': True, 'runEM': False, 'ScaleAndAnalyze': False, 'exportEM': False, 'runNeuron': False, 'plotAll': False}

    
    #MODEL
    #import the images components and build volumes
    if run_parts['importModel']:
        print('import model')
    
        filename = 'CatSpine3_FinalVersion_Aug'
        (model_entities, componentlist) = LoadImages(filename, model_path)
        model_components, elec_params = setParams(componentlist)
        imptime = str(datetime.now()-runtime)
        # print('model components: ', model_components) 
        # print('model entites: ', model_entities) 
        # print('component list: ', componentlist) 
        # print('elec params: ', elec_params)         
 
    #else:
        #if false, none of the following is possible
        #return
    
    if run_parts['placeNeurons']:
        print('Place DTI traces')
        #dti_dir = 'P:/data_raw/cat/scm_task_1/SCS1AC003/DTI/fiber_coord_merged_40mmcoil/FEM-Registered' #os.path.abspath(os.path.join(_CDIR, '..', '..', 'dti_cat3'))
        dti_dir = 'P:/data_raw/cat/scm_task_1/SCS1AC003/DTI/Frank_2022_03 SPARC/4-14-22 Tracts/Tract_coord'
        fiber_names = ['Unmyelinated', 'Small', 'Medium', 'Large']

        fname = 'neuronDist' + runtime.strftime('%Y-%m-%d_%H%M') + '.txt'
        savepath = os.path.join(data_dir, fname)
        # neuron_bodies, dorsal_n = placeNeurons(20, dti_dir, neur_population)
        global neuron_bodies
        global neuron_lbls
        neuron_bodies = placeNeuronsEq(num_neuron, os.path.join(
            dti_dir, 'screened'))  # was running 200/ left or right roots
        placetime = str(datetime.now()-runtime)

        #load neuron file if rerunning same cat
        
        #check if file is present in data folder
        #if so, load it and convert to dictionary
        if 'neurons_'+catname+'.csv' in listdir(data_dir) and loadcat:
            print('Loading neurons for ' + catname)
            # converts back from dictionary to list, use this when loading from csv. 
            ndict = []	
            fpath = os.path.join(data_dir, 'neurons_'+catname+'.csv')
            with open(fpath, mode='r') as file: 
                cf = csv.DictReader(file)
                for line in cf: 
                    ndict.append(dict(line))
            if not savecat: 
                savepath = []
            neuron_lbls = assignNeuronsEq(neuron_bodies, savepath, 0, ndict)

        else: 
            print('Making new set of neurons')
            #if not, run this without that input then save the whole output
            if not savecat: 
                savepath = []
            neuron_lbls = assignNeuronsEq(neuron_bodies, savepath, num_neuron)
            #write to file
            if savecat:
                dia_dict = [{'name': n.name, 'size':n.diameter, 'nerve':n.nerve} for n in neuron_lbls]
                fpath = os.path.join(data_dir, 'neurons_'+catname+'.csv')
                with open(fpath, 'w', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames = dia_dict[0].keys())
                    writer.writeheader()
                    writer.writerows(dia_dict)

        lbltime = str(datetime.now()-runtime)

    elif run_parts['screenNeurons']: #this is elif bc it's only if you're not placing them
        print('Screen DTI traces')
        #dti_dir = 'P:/data_raw/cat/scm_task_1/SCS1AC003/DTI/fiber_coord_merged_40mmcoil/FEM-Registered' #os.path.abspath(os.path.join(_CDIR, '..', '..', 'dti_cat3'))
        dti_dir = 'P:/data_raw/cat/scm_task_1/SCS1AC003/DTI/Frank_2022_03 SPARC/4-14-22 Tracts/Tract_coord'
        fiber_names = ['Unmyelinated', 'Small', 'Medium', 'Large']

        fname = 'neuronDist' + runtime.strftime('%Y-%m-%d_%H%M') + '.txt'
        savepath = os.path.join(data_dir, fname)
        # neuron_bodies, dorsal_n = placeNeurons(20, dti_dir, neur_population)
        # was running 200/ left or right roots
        #TODO: change this part: add timer, change place to screen neurons, add save of data.
        screenNeurons(dti_dir, os.path.join(dti_dir, 'screened'))
        screentime = str(datetime.now()-runtime)
        print('Time to screen neurons ' + screentime)
   
    if run_parts['makeElectrode']:    
    
        print('Generate Electrode Model Volume')
        #This will need to be public (user input)
        #Maria TODO change filepath
        template_dir = os.path.abspath(os.path.join(
            project_path, '..', 'Electrode_Templates_v2/', 'template_' + template + '.sab'))
        xcore.Import(template_dir)

        print('Lead position: ', template, ', contact number: ', contact_n)
        contact = model.AllEntities()['contact '+contact_n]
        contact.Name = 'contact'

        elec_components = [ModelComponent('contact', 'PEC', .5, 5), ModelComponent('silicone', 1e-12, 1, 4)]
        model_components.extend(elec_components)
        electime = str(datetime.now()-runtime)
        # array_start = [3.5, -2, 8] #maria
        # array_start = [-25.8605, 3.1205, 1.446] #IN EP FAT
        #array_start = [-25.8605, 2.8205, 1.446] #IN CSF COMPLETELY
        #array_start = [-25.8605, 2.9505, 1.446] #IN CSF COMPLETELY
        #array_start = [-27.8605, 3.0705, 1.446] 
        #array_start = [-28.3764, 2.4412, 1.446] #LATERAL PLACEMENT ON DURA/EP FAT
        # elec_start = [array_start[0]+ elec_params['elec_offset'][0], array_start[1] + elec_params['elec_offset'][1], array_start[2] + elec_params['elec_offset'][2]]
        
        #New ari
        #elec_start = [-28.0115, 2.51885, 7.73]
        # elec_entities = buildElectrode(elec_params['elec_dim'], elec_params['array_dim'], elec_start, array_start, elec_params['elec_spacing'])
        
        #print('MODEL COMPONENTS ARE: ', model_components)
        #structured grid bits
        #old version (only one surface)
        # elec_sa = elec_params['elec_dim'][1]*elec_params['elec_dim'][2]
        # elec_sa = 2*(elec_params['elec_dim'][1]*elec_params['elec_dim'][2]+elec_params['elec_dim'][1]*elec_params['elec_dim'][0]+elec_params['elec_dim'][0]*elec_params['elec_dim'][2])
    
    if run_parts['WM_Conductivities']:
    
        sim=CreateLFSimulation(project_path)
        sim.RunSimulation(wait=True)
        simname = sim.Name
        CalculateAnisotropicConductivity(simname,document)
        InterpolateAnisotropicConductivity(simname, document)     
        condtime = str(datetime.now()-runtime)
        
    if run_parts['makeStructuredMesh']:
        
        simEM = createStructuredEMSim(model_components)
        simEM.CreateVoxels(project_path)
        
        #Add sphere around the contact
        global tempbox
        tempbox = xcore.GetBoundingBox([model.AllEntities()["contact"]])
        
        #Get the center of the solid
        for i in range(3):
            tempbox[0][i] += elec_params['elec_dim'][i]/2 #Divide by 2 to get the center
         #print(tempbox)
        
        #Build a sphere around it
        flux_sphere = model.CreateSolidSphere(Vec3(tempbox[0]), 1 )
        flux_sphere.Name = 'flux sphere'
        print('Structured mesh complete')
        meshtime = str(datetime.now()-runtime)
    
   
    if run_parts['runEM']:    
        print('run EM sim')
        simEM.RunSimulation(wait=True)    
        print('EM sim complete')
        emtime = str(datetime.now()-runtime)
    
    if run_parts['ScaleAndAnalyze']:
        print('start scaling and analysis')
        ScaleAndAnalyze(simEM)
        print('done scaling and analysis')    
        scaletime = str(datetime.now()-runtime)
    
    #Run EM only results
    if run_parts['runNeuron']:
        #Create and run Neuron simulation
        print('Run Neuron Sim')
        global simN
        global sens_coords
        global elec_coords
        # TODO get this automatically
        elec_coords = [np.mean(
            [tempbox[0][0], tempbox[1][0]]), np.mean([tempbox[0][1], tempbox[1][1]]), np.mean([tempbox[0][2], tempbox[1][2]])]
        #simN, sens_coords = createNeuronDistSim(
        #        neuron_bodies, [elec_start[2]], stim_params, neuron_lbls)
        simN, sens_coords = createNeuronDistSim(
            neuron_bodies, elec_coords, stim_params, neuron_lbls)
        print('Simulation created, running Neuron Simulation')
        ntime1 = str(datetime.now()-runtime)
        simN.RunSimulation(wait=True)
        ntime = str(datetime.now()-runtime)

    #Neuron analysis
    if run_parts['plotAll']:
        print('Extract Analysis')
        global out_v
        global sens_names
        global titrate_vals
        global titrate_names
        #out_v, sens_names = spikes(simN)

        titrate_vals, titrate_names, elec_dist, n_tiss, act_pts, near_pts = titration(simN, elec_coords)
        titratetime = str(datetime.now()-runtime)

        data_dir = os.path.abspath(os.path.join(_CDIR, '..', 'saved_data_finalmodel'))
        #fiber_names.reverse()
        fname = 'titrate' + runtime.strftime('%Y-%m-%d_%H%M') +'.txt'
        savepath = os.path.join(data_dir, fname)
        f = open(savepath, 'w')
        f.write('Electrode Coords: ' + str(elec_coords[0]) + ',' + str(elec_coords[1]) + ',' + str(elec_coords[2]) + '\n')
        f.write('Electrode Dimension: ' + str(elec_params['elec_dim'][0]) + ',' + str(elec_params['elec_dim'][1]) + ',' + str(elec_params['elec_dim'][2]) + '\n')
        f.close()

        fc = open(os.path.join(data_dir, 'elec_opts') + runtime.strftime(
            '%Y-%m-%d_%H%M') + '.csv', 'w', encoding='UTF8', newline='')
        writer = csv.writer(fc)
        #template, contact_n, elec_coords in csv header
        writer.writerow(['model_volumes', 'template_name', 'contact_number', 'neuron_set',
                            'elec_midpt_x', 'elec_midpt_y', 'elec_midpt_z'])
        writer.writerow([filename, template, contact_n, catname] + elec_coords)
        fc.close()

        plotTitration(titrate_vals, titrate_names, neuron_lbls,
                        sens_coords, elec_dist, n_tiss, act_pts, near_pts, savepath)

        plottime = str(datetime.now()-runtime)

        print('Import ' + imptime)
        print('Place N ' + placetime)
        print('Label N ' + lbltime)
        print('Electrode ' + electime)
        print('Conductivities ' + condtime)
        print('Mesh ' + meshtime)
        print('EM ' + emtime)
        print('Scale ' + scaletime)
        print('Load Neuron ' + ntime1)
        print('Run Neuron ' + ntime)
        print('Titrate ' + titratetime)
        print('Plot ' + plottime)


        fname = 'recruit' + runtime.strftime('%Y-%m-%d_%H%M') + '.txt'
        savepath = os.path.join(data_dir, fname)
        recCurve(neuron_lbls, titrate_vals, titrate_names, savepath, 1500, 25)
        #plotTitration(titrate_vals, titrate_names, neuron_lbls, savepath)
        #exportNeurons(neuron_lbls, titrate_vals, titrate_names, stim_params['current_uA'], 500)
        #plotPotentials(out_v, sens_names, fiber_names) 

    print('Done')
    
    
   
    

def main(data_path=None, project_dir=None):
    """
        data_path = path to a folder that contains data for this simulation (e.g. model files)
        project_dir = path to a folder where this project and its results will be saved
    """
    import sys
    import os
    print("Python ", sys.version)
    print("Running in ", os.getcwd(), "@", os.environ['COMPUTERNAME'])

    # if project_dir is None:
    # project_dir = os.path.expanduser(os.path.join('~', 'scs_modeling', 'cat_model_1') )

    # if not os.path.exists(project_dir):
    # os.makedirs(project_dir)
    project_dir = 'P:\\Users\\ard171\\scs_modeling\\cat_model_3'
    fname = os.path.splitext(os.path.basename(_CFILE))[0] + '.smash'
    project_path = os.path.join(project_dir, fname)    
    
    
    print(project_dir)
    print(project_path)
        
    runProject(project_path, os.path.join(project_dir, 'iSegCleaned'))
    

if __name__ == '__main__':
    main()


