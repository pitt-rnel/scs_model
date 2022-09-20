'''Analysis, plotting and text file export for the neuron and EM simulations.'''
from s4l_v1.model import Vec3, Transform
import sys, os
import csv
import numpy
import matplotlib.pyplot as plt
plt.rc('font',**{'family':'sans-serif','sans-serif':['Arial']})

import random
import s4l_v1.document as document
import s4l_v1.analysis as analysis
import s4l_v1.model as model

from helper_fn import *

def titration(sim, elec_coords):
    '''Extracts titration values and location of first spike from neuron sim'''
    elec_coords = numpy.array(elec_coords)
    vol_names = ['csf', 'dorsalroots', 'ventralroots',
                 'whitematter', 'graymatter', 'epiduralfat', 'dura', 'bone'] 
    vols = [model.AllEntities()[x] for x in vol_names]

    simulation_extractor = sim.Results()  # dictionary of results (I think a dictionary of dictionaries)
    
    # Adding a new SensorExtractor
    sensor_extractor = simulation_extractor["Titration Sensor"]
    document.AllAlgorithms.Add(sensor_extractor)

    # Adding a new TitrationEvaluator
    inputs = [sensor_extractor.Outputs["Titration"]]
    titration_evaluator = analysis.neuron_evaluators.TitrationEvaluator(inputs=inputs)
    titration_evaluator.UpdateAttributes()
    document.AllAlgorithms.Add(titration_evaluator)
    titration_evaluator.Update()  # this updates just the algorithm

    nodenames = list(titration_evaluator.LocationOfFirstSpike)
    nrns = list(titration_evaluator.NeuronName)

    #find the location of each of these node (ie inside which volumes?)
    neuron_ents = sim.GetNeuronEntities()
    ent_names = [x.Name for x in neuron_ents]

    elec_dist = [] #distance of the activated node from the electrode
    n_tiss = [] # tissue that each neuron is inside
    act_pts = []
    near_pts = []

    print('Find distance and tissue intersections')
    for i in range(len(nrns)):
        if i%10==0:
            print('Neuron %d of %d' %(i, len(nrns)))
        idx = ent_names.index(nrns[i])
        d = sim.GetSectionGeometries(neuron_ents[idx])

        section_names = sim.GetSectionNames(neuron_ents[idx])
        #TODO: this finds nodes of mrg model, sections of unmyelinated, for other model types figure out correct sections to record
        section_names = [
            x for x in section_names if 'node' in x or 'section' in x]
        # get z coord, seems to be in um
        start_coords = [[d[x].Start[0], d[x].Start[1],
                        d[x].Start[2]] for x in section_names]
        test_sect = section_names[find_nearest3D(elec_coords, start_coords)]
        near_pts.append(d[test_sect].Start)

        st = d[nodenames[i]].Start
        ed = d[nodenames[i]].End
        mdpt = numpy.array([numpy.mean([st[0], ed[0]]), numpy.mean(
            [st[1], ed[1]]), numpy.mean([st[2], ed[2]])])
        act_pts.append(mdpt)

        squared_dist = numpy.sum((mdpt-elec_coords)**2, axis=0)
        elec_dist.append(numpy.sqrt(squared_dist))
        
        #Determine whether a given point is inside a mesh/volume
        e_update = 0
        for j in range(len(vols)):
            #print(i, j)
            dst = model.GetEntityPointDistance(vols[j], Vec3(mdpt)).Distance
            if dst <= 0: #inside a given volume
                n_tiss.append(vol_names[j])
                e_update = 1
                break
        if not e_update: 
            n_tiss.append('undef')
        
        #print(n_tiss)

    return list(titration_evaluator.TitrationFactor), nrns, elec_dist, n_tiss, act_pts, near_pts


def spikes(sim):
    '''Extracts AP shape data from sensors, must be run before plotPotentials'''
    #TODO add any matplotlib stuff later
    # Adding a new SimulationExtractor
    simulation_extractor = sim.Results() #dictionary of results (I think a dictionary of dictionaries)
    
    # Adding the action potential 2D plots
    sensor_v = []
    sensor_names = []
    for sensor in simulation_extractor.keys():
        if "PointSensor" in sensor:
            sensor_extractor = simulation_extractor[sensor]
            #print(sensor_extractor)
            document.AllAlgorithms.Add(sensor_extractor)
            sensor_extractor.Update()
            sensor_v.append(sensor_extractor["v"]) 
            sensor_names.append(sensor_extractor.Name)
            
    plot_viewer = analysis.viewers.PlotViewer(inputs=sensor_v)
    plot_viewer.Update()
    plot_viewer.UpdateAttributes() #for some reason doing this update is necessary to plot things

    sensor_data = []
    for s in range(len(sensor_v)):
        sensor_data = sensor_data + [sensor_v[s].Data.GetComponent(0)]

    #document.AllAlgorithms.Add(plot_viewer)

    return sensor_data, sensor_names

def plotPotentials(sensor_data, sensor_names, fiber_names):
    fig, ax = plt.subplots()

    #xvals = range(len(sensor_data[0]))
    xvals = numpy.linspace(0, 3, len(sensor_data[0]))
    colorm = [[.804, .204, .71], [0, .631, .709], [.082, 0, .776], [1, 0.694, 0.306]]

    for j in range(len(fiber_names)):
        firstplot = True
        for i in range(len(sensor_data)):
            if 'Dorsal' in sensor_names[i]:
                if fiber_names[j] in sensor_names[i]:
                    if firstplot:
                        ax.plot(xvals, sensor_data[i], color=colorm[j], label=fiber_names[j])
                        firstplot = False
                    else:
                        ax.plot(xvals, sensor_data[i], color=colorm[j], label=str())

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.spines['left'].set_position(('data', 0))

    ax.set_title('Example Potentials with 1 mA Stim')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Potential (mV)')
    ax.legend(loc='upper right')


def idx_substring(alist, substring):
    for i, s in enumerate(alist):
        if substring in s:
            return i
    return -1

def plotTitration(sensor_data, sensor_names, neuron_lbls, sens_coords, elec_dist, n_tiss, act_pts, near_pts, savepath):
    '''Makes a scatter plot of titration values based on axon size, and saves all the calculated titration data to a .csv file.'''
    #makes scatter plot of titration values and saves data
    #Todo: could also do this by axon size
    f = open(savepath, 'a')
    fc = open(savepath[:-3]+'csv', 'w', encoding='UTF8', newline='')
    writer = csv.writer(fc)
    writer.writerow(['name', 'nerve', 'diameter', 'elec_dist', 'enc_tissue',
                     'rec_amp', 'act_midpt_x', 'act_midpt_y', 'act_midpt_z', 
                     'closest_x', 'closest_y', 'closest_z'])

    fig, ax = plt.subplots()
    dorsal_data = []
    ventral_data = []
    colorm = [[.804, .204, .71], [0, .631, .709],
              [.082, 0, .776], [1, 0.694, 0.306]]
    fiber_names = ['large', 'medium', 'small', 'unmyelinated'] #if these get renamed in the axon object, change this variable
    fiber_inc = [False, False, False, False]
    sens_coords_keys = list(sens_coords.keys())
    for i in range(len(neuron_lbls)):
        #idx = sensor_names.index(neuron_lbls[i].name)
        idx = idx_substring(sensor_names, neuron_lbls[i].name)
        jitter = random.random()*.8-0.4
        # first plot of this type = label for legend
        if not fiber_inc[fiber_names.index(neuron_lbls[i].type)]:
            j = fiber_names.index(neuron_lbls[i].type)
            #now check if ventral, otherwise space by size
            if not neuron_lbls[i].dorsal:
                ax.plot(4+jitter, sensor_data[idx]*1000,
                        'o', color=colorm[j])
                ventral_data.append(sensor_data[idx]*1000)
            else:
                print(neuron_lbls[i].type)
                fiber_inc[fiber_names.index(neuron_lbls[i].type)] = True
                ax.plot(j+jitter, sensor_data[idx]*1000, 'o',
                        color=colorm[j], label=fiber_names[j])
                dorsal_data.append(sensor_data[idx]*1000)
        else:
            j = fiber_names.index(neuron_lbls[i].type)
            if not neuron_lbls[i].dorsal:
                ax.plot(4+jitter, sensor_data[idx]*1000,
                        'o', color=colorm[j], label=str())
                ventral_data.append(sensor_data[idx]*1000)
            else:
                ax.plot(j+jitter, sensor_data[idx]*1000,
                        'o', color=colorm[j], label=str())
                dorsal_data.append(sensor_data[idx]*1000)
        
        #now save data to file: 
        # FORMAT: Fiber name (includes left/right, dorsal/ventral, level), nerve, size, TODO sensor coordinates, titration amplitude
        # coord_idx = [j for j, x in enumerate(sens_coords_keys) if neuron_lbls[i].name in x]
        # if len(coord_idx)>1:
        #     print('More than one sensor on an axon?')
        # elif len(coord_idx)==0:
        #     print('Index %d named %s not found in coords' %(i, neuron_lbls[i].name))
        # else:
        #     str_coords = str(sens_coords[sens_coords_keys[coord_idx[0]]])
        f.write(neuron_lbls[i].name + ' ' + neuron_lbls[i].nerve_id() + ' of size ' + str(neuron_lbls[i].diameter) + 
            ', dist to recruiting segment ' + str(elec_dist[idx]) + ' recruited in tissue type ' + n_tiss[idx] + ' RecAmp: ' + str(sensor_data[idx]*1000) +
            ' Loc: ' + str(act_pts[idx][0]) + ',' + str(act_pts[idx][1]) + ',' + str(act_pts[idx][2]) + '\n')
            
        writer.writerow([neuron_lbls[i].name, neuron_lbls[i].nerve_id(), neuron_lbls[i].diameter,
                         elec_dist[idx], n_tiss[idx], sensor_data[idx]*1000] + list(act_pts[idx]) + list(near_pts[idx]))
        
    fc.close()
    f.close()

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    print('Ventral mean ' + str(numpy.mean(ventral_data)) +
          ' std dev ' + str(numpy.std(ventral_data)))
    print('Dorsal mean ' + str(numpy.mean(dorsal_data)) +
          ' std dev ' + str(numpy.std(dorsal_data)))

    ax.set_title('Titration of First Recruitment')
    ax.set_xlabel('Axon Type')
    ax.set_ylabel('Recruitment Amplitude (uA)')
    ax.legend(loc='upper left')
    plt.xticks([0, 1, 2, 3, 4], ['A alpha', 'A beta',
                                 'A delta', 'C fiber', 'Ventral'], rotation=0)
    #plt.ylim(0, 1500)
    figpath = os.path.join(savepath, '..', 'titration' + savepath[-17:-4] +'.png')
    plt.savefig(figpath)




def plotEM(resultsPath):
    #import data and sort by z
    file1 = open(resultsPath, 'r')
    data = file1.readlines()
    #line 1 is header
    zcoords = []
    v = []
    em = []
    for i in range(1, len(data)):
        templist = data[i][0:-1].split(',')
        templist = [float(i) for i in templist]
        zcoords.append(templist[2])
        v.append(templist[3])
        em.append(templist[4])
    
    idx = numpy.argsort(zcoords)
    sorted_v = [v[i] for i in idx]
    sorted_em = [em[i] for i in idx]
    sorted_z = [zcoords[i]*1000 for i in idx]

    #calculate upper/lower boundary
    stepsize = 500
    steps = range(0, len(sorted_v), stepsize)
    up = []
    lo = []
    upem = []
    loem = []
    for i in range(len(steps)-1):
        up.append(max(sorted_v[steps[i]:steps[i+1]]))
        lo.append(min(sorted_v[steps[i]:steps[i+1]]))
        upem.append(max(sorted_em[steps[i]:steps[i+1]]))
        loem.append(min(sorted_em[steps[i]:steps[i+1]]))

    #plot the envelope
    fig, ax = plt.subplots()
    zplot = list_idx(sorted_z, steps[0:-1])
    ax.plot(zplot, up)
    ax.plot(zplot, lo)
    # ax.plot(zplot, upem)
    # ax.plot(zplot, loem)
    box_off(ax)
    reset_ticks(ax, [min(zplot), max(zplot)], [min(lo), max(up)])
    ax.set_title('EM field')
    ax.set_xlabel('Z coord (mm)')


def exportNeurons(neuron_lbls, sensor_data, sensor_names, thresh_amp, test_amp):
    '''Takes a list of Axon objects, then based on those names and labels, defines recruitment of neurons types. 
    Sensor data and names should be from the titration data.
    Threshold amplitude should be set to the value of stimulation at which to determine stimulation recruitment.
    For export we need percent of pelvic afferent activation, percent of pudendal afferent activation, and percent of pelvic efferent activation.'''

    pelAffAct = []
    pudAffAct = []
    pelEffAct = []
    sciAffAct = []

    for n in range(len(neuron_lbls)):
        #find the element in the sensor data list
        #idx = sensor_names.index(neuron_lbls[n].name)
        idx = idx_substring(sensor_names, neuron_lbls[n].name)

        #determine if the amplitude of recruitment here is below the stim threshold set
        recruitedBool = int(sensor_data[idx]*thresh_amp < test_amp)

        #for each of the nerves, find all of the nerve
        if neuron_lbls[n].nerve == 1:
            #pelvic nerve
            #dorsal
            if neuron_lbls[n].dorsal:
                pelAffAct.append(recruitedBool)
            #ventral
            else:
                pelEffAct.append(recruitedBool)
        elif neuron_lbls[n].nerve == 2:
            #pudendal nerve - dorsal only needed
            if neuron_lbls[n].dorsal:
                pudAffAct.append(recruitedBool)
        elif neuron_lbls[n].nerve ==3:
            if neuron_lbls[n].dorsal: 
                sciAffAct.append(recruitedBool)

        #print(neuron_lbls[n].nerve_id() + ' ' + str(sensor_data[n]))

    pelAffAct = numpy.mean(pelAffAct)
    pelEffAct = numpy.mean(pelEffAct)
    pudAffAct = numpy.mean(pudAffAct)
    sciAffAct = numpy.mean(sciAffAct)

    #these outputs are percent of the fibers IN THE NERVE not in a root
    # print('Percent pelvic afferents active: ' + str(pelAffAct))
    # print('Percent pudendal afferents active: ' + str(pudAffAct))
    # print('Percent pelvic efferents (SPN) active: ' + str(pelEffAct))
    # print('Other outputs not critical for Task 3')
    return pelAffAct, pudAffAct, pelEffAct, sciAffAct


def recCurve(neuron_lbls, titrate_vals, titrate_names, savepath, thresh_amp, resolution):
    '''Given existing titration data and neuron labels, calculate recruitment curves for pelvic aff and eff, pudendal aff.'''
    pelAffLst = []
    pudAffLst = []
    pelEffLst = []
    sciAffLst = []
    for test_amp in range(0, thresh_amp, resolution):
        pelAff, pudAff, pelEff, sciAff = exportNeurons(neuron_lbls, titrate_vals, titrate_names, thresh_amp, test_amp)
        # if all([pelAff==1, pudAff==1, pelEff==1]):
        #     break
        pelAffLst.append(pelAff)
        pudAffLst.append(pudAff)
        pelEffLst.append(pelEff)
        sciAffLst.append(sciAff)

    #plotting
    fig, ax = plt.subplots()
    xvals = range(0, test_amp+1, resolution)
    print(xvals)
    print(pelAffLst)
    ax.plot(xvals, pelAffLst, label='Pel Aff')
    ax.plot(xvals, pudAffLst, label='Pud Aff')
    ax.plot(xvals, pelEffLst, label='Pel Eff')
    box_off(ax)
    set_corner(ax, 0, 0)
    ax.set_title('Recruitment Curve')
    ax.legend(loc='best')
    ax.set_xlabel('Amplitude (uA)')
    ax.set_ylabel('Proportion Recruited')

    #save to file
    f = open(savepath, 'w')
    str_amps = [str(x) for x in xvals]
    f.write('Amps ' + ",".join(str_amps) + '\n')
    str_pela = [str(round(y, 5)) for y in pelAffLst]
    f.write('PelvicAff ' + ",".join(str_pela) +'\n')
    str_pele = [str(round(y, 5)) for y in pelEffLst]
    f.write('PelvicEff ' + ",".join(str_pele) +'\n')
    str_puda = [str(round(y, 5)) for y in pudAffLst]
    f.write('PudendalAff ' + ",".join(str_puda) +'\n')
    str_scia = [str(round(y, 5)) for y in sciAffLst]
    f.write('SciaticAff ' + ",".join(str_scia) +'\n')

    figpath = os.path.join(savepath, '..', 'rec_curve' + savepath[-17:-4] +'.png')
    plt.savefig(figpath)
