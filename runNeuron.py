'''Functions to run neuron simulation. Neurons must have been named in a format that 
lists level (L7, S1, S2, S3) first and includes Dorsal/Ventral in the name. All neurons must
have unique names.'''

import numpy
import matplotlib.pyplot as plt
plt.rc('font',**{'family':'sans-serif','sans-serif':['Arial']})

import s4l_v1.document as document
import s4l_v1.model as model
import s4l_v1.units as units
import s4l_v1.simulation.neuron as neuron

from helper_fn import *

def createNeuronDistSim(allNeurons, sensor_coords, stim_params, fiber_parameters):
    '''Input a list of spline entities to be added as neurons, a list of z coordinates in 
    mm at which the sensors should be located, parameters of stimulation, fiber info including dorsal/ventral
    and diameter designations. Return simulation object and the list of sensor distances as a dictionary with fiber name'''

    # Create a Neuron simulation
    sim = neuron.Simulation()
    sim.Name = "Neuronal Dynamics"
    #document.AllSimulations.Add(sim)
    entities = model.AllEntities()

    # Setup Settings
    setup_settings = sim.SetupSettings
    #setup_settings.GlobalTemperature = 37.0
    setup_settings.PerformTitration = True
    setup_settings.DepolarizationDetection = setup_settings.DepolarizationDetection.enum.Threshold
    setup_settings.DepolarizationThreshold = 80.0

    # Neuron Settings
    motor_neuron_list = []
    sensory_neuron_list = []
    smallmrg_neuron_list = []
    unmyelinated_neuron_list = []
    #set diameters and sensory/motor based on predefined info in list of Axons
    print('Adding for ' + str(len(fiber_parameters)) + ' total fiber types')
    for i in range(len(fiber_parameters)):
        print(i)
        dia = fiber_parameters[i].diameter
        neuron_body = entities[fiber_parameters[i].name]
        #TODO: right here, add the info
        if dia > 20:
            print('Warning this axon is too large')
        elif dia >= 13:
            if fiber_parameters[i].dorsal:
                print('Adding Large Sensory A-alpha')
                props = model.SensoryMrgNeuronProperties()
                props.AxonDiameter = dia  # default units are microns; allowed range: 4-20 um
                nrn = model.CreateAxonNeuron(neuron_body, props)
                sensory_neuron_list.append(nrn)
            else:
                print('Adding Large Motor Axon')
                props = model.MotorMrgNeuronProperties()
                props.AxonDiameter = dia  # default units are microns; allowed range: 4-20 um
                nrn = model.CreateAxonNeuron(neuron_body, props)
                motor_neuron_list.append(nrn)
        elif dia >= 6: #this is not actually different than the large ones
            if fiber_parameters[i].dorsal:
                print('Adding Medium Sensory A-beta')
                props = model.SensoryMrgNeuronProperties()
                props.AxonDiameter = dia  # default units are microns; allowed range: 4-20 um
                nrn = model.CreateAxonNeuron(neuron_body, props)
                sensory_neuron_list.append(nrn)
            else:
                print('Adding Medium Motor Axon')
                props = model.MotorMrgNeuronProperties()
                props.AxonDiameter = dia  # default units are microns; allowed range: 4-20 um
                nrn = model.CreateAxonNeuron(neuron_body, props)
                motor_neuron_list.append(nrn)

        elif dia >= 2:
            if fiber_parameters[i].dorsal:
                print('Adding A-delta')
                props = model.AdeltaNeuronLTMRProperties()
                props.AxonDiameter = dia  # default units are microns; allowed range: 4-20 um
                nrn = model.CreateAxonNeuron(neuron_body, props)
                sensory_neuron_list.append(nrn)
            else:
                print('Adding small MRG')
                props = model.SmallMrgNeuronProperties()
                props.AxonDiameter = dia  # default units are microns; allowed range: 4-20 um
                nrn = model.CreateAxonNeuron(neuron_body, props)
                smallmrg_neuron_list.append(nrn)
            #TODO need this option
        else:
            print('Adding Unmyelinated C fiber - Note name of settings has changed')
            props = model.SundtNeuronProperties()
            props.AxonDiameter = dia  # default units are microns; allowed range: 4-20 um
            nrn = model.CreateAxonNeuron(neuron_body, props)
            unmyelinated_neuron_list.append(nrn)
    
    # Adding a new AutomaticAxonNeuronSettings
    for axon_list in [sensory_neuron_list, motor_neuron_list, smallmrg_neuron_list, unmyelinated_neuron_list]:
        automatic_axon_neuron_settings = neuron.AutomaticAxonNeuronSettings()
        sim.Add(automatic_axon_neuron_settings, axon_list)

    sim, sens_dict = loadModel(sim, allNeurons, sensor_coords, stim_params) 
    return sim, sens_dict


def loadModel(sim, allNeurons, sensor_coords, stim_params):
    # Discretize the model
	
	# get only the sensor z coordinates
    sensor_zcoords = [sensor_coords[2]]
    #sensor_coords = [x*1000 for x in sensor_coords]

    vol_names = ['csf', 'dorsalroots', 'ventralroots',
                 'whitematter', 'graymatter', 'epiduralfat', 'dura', 'bone']
    vols = [model.AllEntities()[x] for x in vol_names]

    #Get EM info (coincident in model)
    incident_sim = document.AllSimulations["EM Simulation"]
    source_settings = sim.AddSource(incident_sim, "Overall Field")
    #Source Settings! TODO None of these have any info about allowed fields
    # "Dipole" actually means it's a biphasic pulse
    source_settings.PulseType = source_settings.PulseType.enum.Bipolar
    source_settings.InitialTime = stim_params['stim_start_ms']  # ms
    # this number scales the curent density function
    source_settings.AmplitudeP1 = 1.0
    source_settings.DurationP1 = stim_params['stim_pw_ms']  # ms
    source_settings.Interval = 6.6e-05, units.Seconds

    #Sensor settings
    #sections = [d[x] for x in section_names] #To get the Bodies of the sections
    print('Place sensors')
    print(len(allNeurons))
    neuron_ents = sim.GetNeuronEntities()
    print(neuron_ents)
    sens_dict = {}
    rm_ct = 0
    for sens_z in sensor_zcoords: #if placing sensors at multiple z levels - not really necessary usually
        for n in range(len(neuron_ents)):
            print(n)
            #NOTE: because of the order of loading above the order of allNeurons != order of entities below
            fiber = neuron_ents[n]

            #Section Geometries seem to be returned in a dict ordered according to z value
            # this returns a dictionary; {name str: Body}
            d = sim.GetSectionGeometries(fiber)
            section_names = sim.GetSectionNames(fiber)
            #TODO: this finds nodes of mrg model, sections of unmyelinated, for other model types figure out correct sections to record
            section_names = [
                x for x in section_names if 'node' in x or 'section' in x]
            # get z coord, seems to be in um
            start_coords = [[d[x].Start[0], d[x].Start[1], d[x].Start[2]] for x in section_names]

            test_sect = section_names[find_nearest3D(sensor_coords, start_coords)]
            print(test_sect + ' at z position ' +
                  str(d[test_sect].Start[2]) + ' mm')
            #print(d[test_sect].Start)

            #ignore sensors if they are very far away
            # print('TODO: instead of doing distance maybe check if sensor is in saline/bone')
            dist_to_sensor = numpy.sqrt(
                ((d[test_sect].Start[0]-sensor_coords[0])**2)+((d[test_sect].Start[1]-sensor_coords[1])**2)+((d[test_sect].Start[2]-sensor_coords[2])**2))
            if dist_to_sensor > 40:
                #TROUBLESHOOTING TODO
                print('Ignore ' + fiber.Name +
                        ' with long sensor distance: ' + str(dist_to_sensor))
                # print(d[test_sect].Start)
                # print(sensor_coords)
                #actually delete the neuron - reduce processing time, I think
                neuron_ents[n].Delete()
                
                # for j in range(len(vols)):
                #     dst = model.GetEntityPointDistance(
                #         vols[j], d[test_sect].Start).Distance
                #     print(dst)
                #     if dst <= 0:  # inside a given volume
                #         print(vol_names[j])
                #         break

                rm_ct+=1
                # print('idx ' + str(n) + ' rm_ct at '+ str(rm_ct))
            else:
                print('Include ' + fiber.Name)
                #redefine point sensor settings for each one, fine to overwrite
                point_sensor_settings = sim.AddPointSensor(fiber)
                point_sensor_settings.SectionName = test_sect
                point_sensor_settings.RelativePosition = 0.5  # halfway along the section
                #add sensor name, coordinates to a dictionary
                sens_dict[fiber.Name + '_'  + test_sect] = list(d[test_sect].Start)

    print('Ignored ' + str(rm_ct) + ' neurons')
    #TODO again, what does any of this mean
    #I think this is explained in neuron.SolverSettings
    solver_settings = sim.SolverSettings
    solver_settings.Duration = 0.003, units.Seconds
    solver_settings.TimeStep = 2e-06, units.Seconds
    solver_settings.NumberOfThreads = 10
    document.AllSimulations.Add(sim)

    return sim, sens_dict
