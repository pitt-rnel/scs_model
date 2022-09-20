'''All functions and dependencies to run the import of DTI traces and neuron labeling from file'''

import os
from posixpath import splitdrive
import matplotlib.pyplot as plt
from numpy import save
plt.rc('font',**{'family':'sans-serif','sans-serif':['Arial']})
from numpy.lib.npyio import save, zipfile_factory
import numpy
#import pandas as pd
import random
import math 
import s4l_v1.model as model
import XCoreModeling as xcore
from s4l_v1.model import Vec3
import copy

from helper_fn import * 

def placeNeurons(num_neurons, dti_dir):
    '''Given a total number of neurons, this function will distribute them between the L6-S3 roots, calculate the 
    sizes to use, name the neurons. It is CRITICAL that DTI files are named with the first two characters indicating
    location, and if they are separated into right and left they must include an underscore and l or r (ie "L6_r.txt")'''
    #num_neurons is number of neurons to pull from the DTI file per spinal level; a good default is 10-20
    #dti_dir is a directory containing only the text files with dti information for all levels
    #neur_population is a 2D list, with the first item being a list of neuron name strings, and the second
    # being an equal-length list of the proportion/percent of each of those types for the random draw
    dti_files = os.listdir(dti_dir)
    vol_names = ['dorsalroots', 'ventralroots']
    root_vols = [model.AllEntities()[x] for x in vol_names]
    #csf_body = model.AllEntities()['CSF']

    #NEURON DISTRIBUTIONS (revised March 2022 - info in Model Neuron distributions ppt and model_distributions.m)
    #NEURON DISTRIBUTIONS FOR BLIND DIST - relevant neurons per level (sums to 1)
    v_rel = {'L6': 0.0339, 'L7': 0.0587, 'S1': 0.0753, 'S2': 0.0433, 'S3': 0.0286}
    d_rel = {'L6': 0.1602, 'L7': 0.2795, 'S1': 0.1610, 'S2': 0.1284, 'S3': 0.0312}

    #Calculate the number of axons per level, given the input total number of neurons
    v_rel.update((x, round(y*num_neurons)) for x, y in v_rel.items())
    d_rel.update((x, round(y*num_neurons)) for x, y in d_rel.items())

    dti_files = [x for x in dti_files if x[0:2] in v_rel.keys()] #Don't bother placing neurons if we aren't assigning them to anything

    # ntype_count = [0 for x in neur_population[0]]
    root_count = 0
    neuronlist = []
    #dorsal_n = []
    #use root_clrs if you want to visualize all of the neurons from different roots as different colors
    root_clrs = {'L5': [1, 1, .57], 'L6': [1, .73, .45], 'L7': [1, .65, .82], 'S1': [.8, .62, 1], 'S2': [.55, .7, 1], 'S3': [.55, .9, 1]}
    for i in range(len(dti_files)):
        #tempcolor = xcore.Color(root_clrs[dti_files[i][0:2]][0], root_clrs[dti_files[i][0:2]][1], root_clrs[dti_files[i][0:2]][2], 1)
        tempcolor = xcore.Color(1, 1, 1, 1)
        
        #check total number of axons + number needed for both dorsal/ventral
        num_dorsal = d_rel[dti_files[i][0:2]]
        num_ventral = v_rel[dti_files[i][0:2]]
        #if these are split into left and right side files, halve the amount of neurons placed per file
        if '_r' in dti_files[i].lower():
            num_dorsal_r = math.ceil(num_dorsal/2)
            num_ventral_r = math.ceil(num_ventral/2)
            num_dorsal_l = 0
            num_ventral_l = 0
        elif '_l' in dti_files[i].lower():
            num_dorsal_l = math.ceil(num_dorsal/2)
            num_ventral_l = math.ceil(num_ventral/2)
            num_dorsal_r = 0
            num_ventral_r = 0
        #if the file isn't split, still allot how many per side
        else:
            num_dorsal_l = math.ceil(num_dorsal/2)
            num_ventral_l = math.ceil(num_ventral/2)
            num_dorsal_r = math.ceil(num_dorsal/2)
            num_ventral_r = math.ceil(num_ventral/2)
        
        print('Loading from file ' + dti_files[i])
        print(num_dorsal_l)
        print(num_dorsal_r)
        print(num_ventral_l)
        print(num_ventral_r)

        fpath = os.path.join(dti_dir, dti_files[i])
        f = open(fpath, 'r')
        fulltext = f.read()

        splittext = fulltext.splitlines()
        # linect = str(len(splittext))
        # range(len(splittext)): # for now not whole range range(len(splittext)):

        for j in range(len(splittext)):
            #print('Load line ' + str(j) + ' of ' + linect)
            teststr = splittext[j]
            float_coords = str2coords(teststr)

            if len(float_coords) > 10: #only if it's a long trace
                #determine whether it's dorsal or ventral
                #dorsal_n = checkDV(float_coords, root_vols)
                dorsal_n = '_d' in dti_files[i] #this line requires that files are pre-screened and split into dorsal and ventral files
                # if dorsal_n==-1:
                #     #not in dorsal or ventral roots at midpoint
                #     continue
                #check if it's left or right side
                cutoff = 13
                lr_lbl = checkLR(float_coords, cutoff, dti_files[i])
                
                if dorsal_n and 'l' in lr_lbl.lower() and num_dorsal_l-1<0:
                    #skip this neuron if all left dorsal ones already added
                    continue
                elif dorsal_n and 'r' in lr_lbl.lower() and num_dorsal_r-1 < 0:
                    #skip this neuron if all right dorsal ones added
                    continue
                elif not dorsal_n and 'l' in lr_lbl.lower() and num_ventral_l-1 < 0:
                    #skip this neuron if all left ventral ones added
                    continue
                elif not dorsal_n and 'r' in lr_lbl.lower() and num_ventral_r-1 < 0:
                    #skip this neuron if all right ventral ones added
                    continue
                
                #make labels
                if dorsal_n:
                    nname = dti_files[i][0:2] + lr_lbl + ' Dorsal Neuron ' + \
                        str(root_count)
                else:
                    nname = dti_files[i][0:2] + lr_lbl + ' Ventral Neuron ' + \
                        str(root_count)
                
                #make splines
                tempn = makeNeuron(float_coords, nname)
                if tempn is not None:
                    #sometimes the line above returns without adding a neuron; only increment when it does
                    neuronlist.append(tempn)
                    
                    root_count += 1 #increment number used to uniquely label neurons
                    neuronlist[-1].Color = tempcolor #set the neuron color in the display
                    #update the counter for whichever case: 
                    if dorsal_n and 'l' in lr_lbl.lower():
                        num_dorsal_l-=1
                    elif dorsal_n and 'r' in lr_lbl.lower():
                        num_dorsal_r-=1
                    elif ~dorsal_n and 'l' in lr_lbl.lower():
                        num_ventral_l-=1
                    elif ~dorsal_n and 'r' in lr_lbl.lower():
                        num_ventral_r-=1
                
                if num_dorsal_l == 0 and num_dorsal_r == 0 and num_ventral_l == 0 and num_ventral_r == 0:
                    #quit adding neurons if the total number set above has been added
                    #print('break loop')
                    break
            else:
                print('skipped short trace')
                continue
        print(num_dorsal_l)
        print(num_dorsal_r)
        print(num_ventral_l)
        print(num_ventral_r)

    return neuronlist


def placeNeuronsEq(num_neurons, dti_dir):
    '''Given a total number of neurons, this function will distribute them EQUALLY between the L6-S3 roots, calculate the 
    sizes to use, name the neurons. It is CRITICAL that DTI files are named with the first two characters indicating
    location, and if they are separated into right and left they must include an underscore and l or r (ie "L6_r.txt")'''
    #num_neurons is number of neurons to pull from the DTI file per spinal level; a good default is 10-20
    #dti_dir is a directory containing only the text files with dti information for all levels
    #neur_population is a 2D list, with the first item being a list of neuron name strings, and the second
    # being an equal-length list of the proportion/percent of each of those types for the random draw
    dti_files = os.listdir(dti_dir)
    vol_names = ['dorsalroots', 'ventralroots']
    root_vols = [model.AllEntities()[x] for x in vol_names]
    #csf_body = model.AllEntities()['CSF']

    #NEURON DISTRIBUTIONS - ALL EQUAL; number of neurons * number of nerves from that root * left/right
    v_rel = {'L6': num_neurons*2, 'L7': num_neurons*2,
             'S1': num_neurons*3*2, 'S2': num_neurons*3*2, 'S3': num_neurons*2*2}
    d_rel = {'L6': num_neurons*2, 'L7': num_neurons*2,
             'S1': num_neurons*3*2, 'S2': num_neurons*3*2, 'S3': num_neurons*2*2}

    # Don't bother placing neurons if we aren't assigning them to anything
    dti_files = [x for x in dti_files if x[0:2] in v_rel.keys()]

    # ntype_count = [0 for x in neur_population[0]]
    root_count = 0
    neuronlist = []
    #dorsal_n = []
    #use root_clrs if you want to visualize all of the neurons from different roots as different colors
    root_clrs = {'L5': [1, 1, .57], 'L6': [1, .73, .45], 'L7': [
        1, .65, .82], 'S1': [.8, .62, 1], 'S2': [.55, .7, 1], 'S3': [.55, .9, 1]}
    for i in range(len(dti_files)):
        #tempcolor = xcore.Color(root_clrs[dti_files[i][0:2]][0], root_clrs[dti_files[i][0:2]][1], root_clrs[dti_files[i][0:2]][2], 1)
        tempcolor = xcore.Color(1, 1, 1, 1)

        #check total number of axons + number needed for both dorsal/ventral
        num_dorsal = d_rel[dti_files[i][0:2]]
        num_ventral = v_rel[dti_files[i][0:2]]
        #if these are split into left and right side files, halve the amount of neurons placed per file
        if '_r' in dti_files[i].lower():
            num_dorsal_r = math.ceil(num_dorsal/2)
            num_ventral_r = math.ceil(num_ventral/2)
            num_dorsal_l = 0
            num_ventral_l = 0
        elif '_l' in dti_files[i].lower():
            num_dorsal_l = math.ceil(num_dorsal/2)
            num_ventral_l = math.ceil(num_ventral/2)
            num_dorsal_r = 0
            num_ventral_r = 0
        #if the file isn't split, still allot how many per side
        else:
            num_dorsal_l = math.ceil(num_dorsal/2)
            num_ventral_l = math.ceil(num_ventral/2)
            num_dorsal_r = math.ceil(num_dorsal/2)
            num_ventral_r = math.ceil(num_ventral/2)

        print('Loading from file ' + dti_files[i])
        print(num_dorsal_l)
        print(num_dorsal_r)
        print(num_ventral_l)
        print(num_ventral_r)

        fpath = os.path.join(dti_dir, dti_files[i])
        f = open(fpath, 'r')
        fulltext = f.read()

        splittext = fulltext.splitlines()
        # linect = str(len(splittext))
        # range(len(splittext)): # for now not whole range range(len(splittext)):

        for j in range(len(splittext)):
            #print('Load line ' + str(j) + ' of ' + linect)
            teststr = splittext[j]
            float_coords = str2coords(teststr)

            if len(float_coords) > 10:  # only if it's a long trace
                #determine whether it's dorsal or ventral
                #dorsal_n = checkDV(float_coords, root_vols)
                # this line requires that files are pre-screened and split into dorsal and ventral files
                dorsal_n = '_d' in dti_files[i]
                # if dorsal_n==-1:
                #     #not in dorsal or ventral roots at midpoint
                #     continue
                #check if it's left or right side
                cutoff = 13
                lr_lbl = checkLR(float_coords, cutoff, dti_files[i])

                if dorsal_n and 'l' in lr_lbl.lower() and num_dorsal_l-1 < 0:
                    #skip this neuron if all left dorsal ones already added
                    continue
                elif dorsal_n and 'r' in lr_lbl.lower() and num_dorsal_r-1 < 0:
                    #skip this neuron if all right dorsal ones added
                    continue
                elif not dorsal_n and 'l' in lr_lbl.lower() and num_ventral_l-1 < 0:
                    #skip this neuron if all left ventral ones added
                    continue
                elif not dorsal_n and 'r' in lr_lbl.lower() and num_ventral_r-1 < 0:
                    #skip this neuron if all right ventral ones added
                    continue

                #make labels
                if dorsal_n:
                    nname = dti_files[i][0:2] + lr_lbl + ' Dorsal Neuron ' + \
                        str(root_count)
                else:
                    nname = dti_files[i][0:2] + lr_lbl + ' Ventral Neuron ' + \
                        str(root_count)

                #make splines
                tempn = makeNeuron(float_coords, nname)
                if tempn is not None:
                    #sometimes the line above returns without adding a neuron; only increment when it does
                    neuronlist.append(tempn)

                    root_count += 1  # increment number used to uniquely label neurons
                    # set the neuron color in the display
                    neuronlist[-1].Color = tempcolor
                    #update the counter for whichever case:
                    if dorsal_n and 'l' in lr_lbl.lower():
                        num_dorsal_l -= 1
                    elif dorsal_n and 'r' in lr_lbl.lower():
                        num_dorsal_r -= 1
                    elif ~dorsal_n and 'l' in lr_lbl.lower():
                        num_ventral_l -= 1
                    elif ~dorsal_n and 'r' in lr_lbl.lower():
                        num_ventral_r -= 1

                if num_dorsal_l == 0 and num_dorsal_r == 0 and num_ventral_l == 0 and num_ventral_r == 0:
                    #quit adding neurons if the total number set above has been added
                    #print('break loop')
                    break
            else:
                print('skipped short trace')
                continue
        print(num_dorsal_l)
        print(num_dorsal_r)
        print(num_ventral_l)
        print(num_ventral_r)

    return neuronlist

def screenNeurons(dti_dir, save_dir):
    '''Given a total number of neurons, this function will distribute them between the L6-S3 roots, calculate the 
    sizes to use, name the neurons. It is CRITICAL that DTI files are named with the first two characters indicating
    location, and if they are separated into right and left they must include an underscore and l or r (ie "L6_r.txt")'''
    #num_neurons is number of neurons to pull from the DTI file per spinal level; a good default is 10-20
    #dti_dir is a directory containing only the text files with dti information for all levels
    #neur_population is a 2D list, with the first item being a list of neuron name strings, and the second
    # being an equal-length list of the proportion/percent of each of those types for the random draw
    dti_files = os.listdir(dti_dir)
    print('Updated5')
    vol_names = ['dorsalroots', 'ventralroots', 'graymatter', 'whitematter']
    root_vols = [model.AllEntities()[x] for x in vol_names]
    #csf_body = model.AllEntities()['CSF']
    cutoff_zvals = {'L6':47.6, 'L7':33.2, 'S1':26.6, 'S2':22.6, 'S3':18.6}
 
    for i in range(9,len(dti_files)): #for testing
        if dti_files[i]=='screened':#ignore folder
            continue
        print(dti_files[i])
        fpath = os.path.join(dti_dir, dti_files[i])
        f = open(fpath, 'r')
        fulltext = f.read()

        splittext = fulltext.splitlines()
        # linect = str(len(splittext))
        # range(len(splittext)): # for now not whole range range(len(splittext)):
        not_skip_n = []
        d_n = []
        v_n = []
        dns = []
        vns = []
        vnu = []
        all_u = []
        for j in range(len(splittext)): #screen all neurons
            #assume the last value in the list is most rostral
            #print('Load line ' + str(j) + ' of ' + linect)
            print(dti_files[i])
            # print(cutoff_zvals[dti_files[i][0:2]])
            teststr = splittext[j]
            float_coords = str2coords_convert(teststr)
            #print(float_coords)
            zv = [idx for idx, x in enumerate(float_coords) if x[2]<=cutoff_zvals[dti_files[i][0:2]]]
            if len(zv)>0:
                # print(float_coords[zv[-1]])
                # print(float_coords[-1])
                # print(float_coords[0:zv[3]])
                float_coords = float_coords[zv[-1]:-1]

            if len(float_coords) > 5:
                # len(float_coords) > 10:  # only if it's a long trace
                if abs(float_coords[0][2]-float_coords[-1][2]) > 20:
                    #determine whether it's dorsal or ventral
                    dorsal_n = checkDV(float_coords, root_vols)
                    if dorsal_n == -1:
                        #not in dorsal or ventral roots at midpoint
                        #make splines
                        #temporary version, for s3l temp:
                        # at z=40, y <9 
                        # ventral
                        if float_coords[find_nearest(40, [x[2] for x in float_coords])][1]<9:
                            vnu.append(coord2str(float_coords) + '\n')
                        tempn = makeNeuron(float_coords, 'uncontained')
                        if tempn is not None:
                            tempn.Color = xcore.Color(1, 0, 0, 1)
                            #update the counter for whichever case:
                        print('skip')
                        all_u.append(coord2str(float_coords) + '\n')
                        continue
                    elif dorsal_n:
                        d_n.append(coord2str(float_coords) + '\n')
                        #make splines
                        tempn = makeNeuron(float_coords, 'neuron')
                        if tempn is not None:
                            tempn.Color = xcore.Color(1, 1, 1, 1)
                            #update the counter for whichever case:
                    else:
                        v_n.append(coord2str(float_coords) + '\n')
                    not_skip_n.append(j)

                    
                else:
                    dorsal_n = checkDV(float_coords, root_vols)
                    if dorsal_n == -1:
                        #not in dorsal or ventral roots at midpoint
                        #make splines
                        # tempn = makeNeuron(float_coords, 's uncontained')
                        # if tempn is not None:
                        #     tempn.Color = xcore.Color(0, 1, 0, 1)
                        #update the counter for whichever case:
                        print('skip')
                        continue
                    elif dorsal_n:
                        dns.append(coord2str(float_coords) + '\n')
                    else:
                        vns.append(coord2str(float_coords) + '\n')

                    #make splines
                    tempn = makeNeuron(float_coords, 'short')
                    if tempn is not None:
                        tempn.Color = xcore.Color(0, 0, 1, 1)
                    #update the counter
                    #  for whichever case:
                    print('skipped short')

            else:
                #make splines
                tempn = makeNeuron(float_coords, 'supershort')
                if tempn is not None:
                    tempn.Color = xcore.Color(1, 0, 0, 1)
                    #update the counter for whichever case:
                # else:
                #     print(float_coords)
                print('skipped short trace')
                continue


        #TODO save everything not labeled skip to a txt file, same format
        print(os.path.join(save_dir, dti_files[i]))
        fs = open(os.path.join(save_dir, dti_files[i][:-4]+'_d'+dti_files[i][-4:]), 'a')
        fs.writelines(d_n)
        fs.close()
        fs = open(os.path.join(
            save_dir, dti_files[i][:-4]+'_v'+dti_files[i][-4:]), 'a')
        fs.writelines(vnu)
        fs.close()
        
    return d_n, v_n, dns, vns, all_u


def assignNeuronsEq(neuronlist, savepath, num_neurons, diameterlist = []):
    '''This version of assignNeurons does so with an equal number of neurons in each
    nerve at each level, based on a defined number of neurons. 
    neuronlist should include neurons with unique names that begin with level, 
    and include the word dorsal or ventral (separated by spaces)
    Format of diameterlist is: 
    list of dictionaries as [{'name':neuron name string, 'size':diameter string}]'''

    #for each root level, include neurons for these nerves: 
    nerve_root_inc = {'L6':[0, 0, 1], 'L7':[0, 0, 1], 'S1':[1, 1, 1], 'S2':[1, 1, 1], 'S3':[1, 1, 0]}

    #Typical size distribution for each NERVE - in same order as the v_ratio and d_ratio arrays (pelvic, pudendal, sciatic)
    sizebins = range(1, 21)
    nerve_dists_d = [[0, 55.9, 26.4, 5.9, 5.9, 5.9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #pelvic
                [0, 3.7, 7.4, 10.4, 12.9, 13.9, 13.1, 12.1, 10.2, #pudendal - note c fibers removed
                    8, 4.8, 1.2, 0.9, 0.7, 0.3, 0.4, 0, 0, 0, 0], 
                [0, 0, 0.2, 4.5, 8.9, 10.0, 7.9, 6.9, 4.9, 4.5, 5.2, 5.3, 5.3, 6.7, 7.4, 7.3, 6.6, 4.8, 2.5, 1.1]] #sciatic
    nerve_dists_v = [[0, 41.7, 50.9, 7.4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 4.6, 4.9, 11.5, 4.6, 5.9, 17.5, 13.2,
                    9, 6.2, 8.2, 0.1, 4.4, 0.9, 4.6, 4.4, 0, 0],
                [0, 0, 0.2, 4.5, 8.9, 10.0, 7.9, 6.9, 4.9, 4.5, 5.2, 5.3, 5.3, 6.7, 7.4, 7.3, 6.6, 4.8, 2.5, 1.1]]

    v_actual = [[], [], []]
    d_actual = [[], [], []]
    #pull out names from the neuronlist
    nnames = [n.Name for n in neuronlist]
    if len(diameterlist)==0:
        make_dlist = True
        diameterlist = []
    else: 
        make_dlist = False
        nmlist = [x['name'] for x in diameterlist]

    neurontypes = []

    #update list to include all the neurons to assign
    nerve_root_inc.update((x, [i*num_neurons for i in y]) for x, y in nerve_root_inc.items())
    nr_inc_ld = copy.deepcopy(nerve_root_inc)
    nr_inc_rd = copy.deepcopy(nerve_root_inc)
    nr_inc_lv = copy.deepcopy(nerve_root_inc)
    nr_inc_rv = copy.deepcopy(nerve_root_inc)

    #cycle through neurons
    for n in neuronlist: 
        print(n.Name)
        tmp = n.Name.split(' ')
        #decide if dorsal or ventral
        aff = 'dorsal' in [i.lower() for i in tmp]
        #label left/right
        leftside = 'left' in [i.lower() for i in tmp]
        #label root
        rootlvl = tmp[0]
        #subtract from the correct nerve + assign

        if make_dlist:
            if tmp[0] not in nerve_root_inc:
                #Skip nerves at this level - no distribution data
                continue
            #for each nerve at a level + aff/eff + left/right, generate num_neuron neurons
            #now pull from the relevant nerve distribution
            if aff: 
                if leftside:
                    #find first nerve that's not filled in the level
                    nnext = next((i for i, x in enumerate(
                        nr_inc_ld[rootlvl]) if x != 0), None)
                    #subtract from that level
                    print(nr_inc_ld)
                    nr_inc_ld[rootlvl][nnext] = nr_inc_ld[rootlvl][nnext]-1
                else: 
                    #find first nerve that's not filled in the level
                    nnext = next((i for i, x in enumerate(
                        nr_inc_rd[rootlvl]) if x != 0), None)
                    #subtract from that level
                    nr_inc_rd[rootlvl][nnext] = nr_inc_rd[rootlvl][nnext]-1
                diameter_um = random.choices(
                    sizebins, nerve_dists_d[nnext])[0]
                #mark the chosen diameter
                d_actual[nnext].append(diameter_um)
            else: 
                if leftside:
                    #find first nerve that's not filled in the level
                    nnext = next((i for i, x in enumerate(
                        nr_inc_lv[rootlvl]) if x != 0), None)
                    #subtract from that level
                    nr_inc_lv[rootlvl][nnext] = nr_inc_lv[rootlvl][nnext]-1
                else: 
                    #find first nerve that's not filled in the level
                    nnext = next((i for i, x in enumerate(
                        nr_inc_rv[rootlvl]) if x != 0), None)
                    #subtract from that level
                    nr_inc_rv[rootlvl][nnext] = nr_inc_rv[rootlvl][nnext]-1
                diameter_um = random.choices(
                    sizebins, nerve_dists_v[nnext])[0]
                #mark the chosen diameter
                v_actual[nnext].append(diameter_um)

        else:  # assign diameters from the input list until you run out of axons
            # find this item in the list; should be first.
            nidx = nmlist.index(n.Name)
            #will error if there's a neuron named something not in the list
            diameter_um = int(diameterlist[nidx]['size'])
            #label this by what nerve it is
            nnext = int(diameterlist[nidx]['nerve'])-1
            if aff:
                d_actual[nnext].append(diameter_um)
            else: 
                v_actual[nnext].append(diameter_um)


        #add class label, for now this is a placeholder but if we add overlap this is important
        if diameter_um > 13:
            fiber_type = 'large'
        elif diameter_um > 5:
            fiber_type = 'medium'
        elif diameter_um > 1:
            fiber_type = 'small'
        else:
            fiber_type = 'unmyelinated'

        #Add neuron type
        neurontypes.append(
            Axon(n.Name, aff, nnext+1, diameter_um, fiber_type, rootlvl))

    #plot the distributions
    if savepath: 
        tmp = savepath.split('.txt')[0]

        nnames = ['pelvic', 'pudendal', 'sciatic']
        plotNDist(dict(zip(nnames, nerve_dists_v)), dict(zip(nnames, v_actual)), numpy.arange(0.5, 21.5, 1), tmp+'v.txt')
        plotNDist(dict(zip(nnames, nerve_dists_d)), dict(
            zip(nnames, d_actual)), numpy.arange(0.5, 21.5, 1), tmp+'d.txt')
    
    #return the list of Axon objects
    return neurontypes

    


def assignNeurons(neuronlist, savepath, diameterlist = []):
    '''neuronlist should include neurons with unique names that begin with level, 
    and include the word dorsal or ventral (separated by spaces)
    for each spinal level, distribution of nerve destinations (pelvic, pudendal, sciatic, other)
    for each nerve, distribution of sizes
    Format of diameterlist is: list of dictionaries as [{'name':neuron name string, 'size':diameter string}]
    '''
    #for nerve distributions, an example input, where each level has an array defining the percent of 
    #each nerve in the root, ordered [pelvic, pudendal, sciatic, other]

    #Full size distributions for each level, based on nerve distributions (this was calculated in the matlab script model_distributions.m)
    vroot_dist = {'L6': [0, 0, 1, 4, 9, 10, 8, 7, 5, 5, 5, 5, 5, 7, 7, 7, 7, 5, 2, 1],
                  'L7': [0, 0, 1, 4, 9, 10, 8, 7, 5, 5, 5, 5, 5, 7, 7, 7, 7, 5, 2, 1],
                  'S1': [0, 3, 4, 5, 6, 10, 5, 6, 11, 10, 7, 5, 7, 2, 5, 3, 5, 4, 1, 1],
                  'S2': [0, 26, 32, 6, 2, 4, 2, 2, 5, 4, 3, 2, 3, 1, 2, 1, 2, 1, 1, 1],
                  'S3': [0, 6, 7, 5, 4, 10, 4, 5, 14, 11, 8, 5, 7, 1, 4, 1, 4, 4, 0, 0]}
    droot_dist = {'L6': [0, 0, 1, 4, 9, 10, 8, 7, 5, 5, 5, 5, 5, 7, 7, 7, 7, 5, 2, 1],
                  'L7': [0, 0, 1, 4, 9, 10, 8, 7, 5, 5, 5, 5, 5, 7, 7, 7, 7, 5, 2, 1],
                  'S1': [0, 6, 4, 6, 9, 10, 8, 7, 6, 5, 5, 4, 4, 5, 5, 5, 5, 3, 2, 1],
                  'S2': [0, 31, 16, 7, 8, 9, 6, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                  'S3': [0, 35, 18, 8, 8, 8, 5, 4, 4, 3, 2, 1, 1, 1, 1, 1, 0, 0, 0, 0]}
    #for diameter distributions within a nerve: binned percentages
    dia_binEdges = range(1, 21) #length of this array must be equal to each distribution
    #note that pelvic and pudendal nerves need these values separated by aff/eff, esp because pelvic tends to have lots of unmyelinated 
    #motor axons, and bc pelvic also has lots of unmyelinated stuff from colon
   
    v_actual = {'L6': [], 
                'L7': [], 
                'S1': [], 
                'S2': [], 
                'S3': []}
    d_actual = {'L6': [],
                'L7': [],
                'S1': [],
                'S2': [],
                'S3': []}
    
    if len(diameterlist)==0:
        make_dlist = True
        diameterlist = []
    else: 
        make_dlist = False
        nmlist = [x['name'] for x in diameterlist]

    neurontypes = []
    
    for n in neuronlist:
        tmp = n.Name.split(' ')
        aff = 'dorsal' in [i.lower() for i in tmp]

        if make_dlist:
            if tmp[0] not in vroot_dist:
                #Skip nerves at this level - no distribution data
                continue
            
            #generate diameter based on weighted probabilities
            if aff:
                diameter_um = random.choices(dia_binEdges, droot_dist[tmp[0]])[0]
                #mark the chosen diameter
                d_actual[tmp[0]].append(diameter_um)
            else:
                diameter_um = random.choices(dia_binEdges, vroot_dist[tmp[0]])[0]
                #mark the chosen diameter
                v_actual[tmp[0]].append(diameter_um)


        else: #assign diameters from the input list until you run out of axons
            nidx = nmlist.index(n.Name) #find this item in the list; should be first. 
            #will error if there's a neuron named something not in the list
            diameter_um = int(diameterlist[nidx]['size']) 

            if aff:
                #mark the chosen diameter
                d_actual[tmp[0]].append(diameter_um)
            else:
                #mark the chosen diameter
                v_actual[tmp[0]].append(diameter_um)

        #add class label, for now this is a placeholder but if we add overlap this is important
        if diameter_um>13:
            fiber_type = 'large'
        elif diameter_um>5:
            fiber_type = 'medium'
        elif diameter_um>1:
            fiber_type = 'small'
        else:
            fiber_type = 'unmyelinated'

        #Add neuron type, but label them all as "other" for now
        neurontypes.append(Axon(n.Name, aff, 0, diameter_um, fiber_type, tmp[0]))

    #plot these distributions
    tmp = savepath.split('.txt')[0]
    plotNDist(vroot_dist, v_actual, numpy.arange(0.5, 21.5, 1), tmp+'v.txt')
    plotNDist(droot_dist, d_actual, numpy.arange(0.5, 21.5, 1), tmp+'d.txt')
    #save diameterlist to file outside of this 

    #return the list of Axon objects
    return neurontypes

def plotNDist(ideal_dist, actual_dist, bins, savepath):
    '''input the ideal distribution as percent values for each of the bins, 
    input the actual distribution as the list of values assigned'''
    f = open(savepath, 'w')
    lbl = savepath.split('neuronDist')[1]
    #print('this is new' + '2')
    allkeys = [x for x in ideal_dist.keys()]
    # print(ideal_dist)
    # print(actual_dist)
    binmid = [(x + y)/2 for (x, y) in zip(bins[:-1], bins[1:])]
    for key in allkeys:
        if len(actual_dist[key])==0:
            continue
        #make a new plot
        fig, ax = plt.subplots()
        #plot actual distributions as a histogram
        plt.hist(actual_dist[key], bins, alpha = 0.6, color=[.24, .64, 1])
        plt.title(key)
        plt.xlabel('Axon size (um)')
        plt.ylabel('Count')

        #plot the ideal distribution as individual points, scaled for the number of neurons in actual dist
        vals = [x*len(actual_dist[key])/100 for x in ideal_dist[key]]
        #print(vals)
        #print(binmid)
        plt.scatter(binmid, vals, marker = 'o', alpha = 0.8, c=[[0, 0, 0]])

        box_off(ax)
        imax = max([max(actual_dist[key]), max(vals)])
        plt.ylim([0, imax*1.2])
        reset_ticks(ax, [bins[0], bins[-1]], [0, imax])

        #save to file
        f.write(key + '\n')
        f.write('bins ' + str(binmid) + '\n')
        f.write('actual ' + str(actual_dist[key]) + '\n')
        f.write('ideal ' + str(vals) + '\n')

        figpath = os.path.join(savepath, '..', key.lower() + '_dist' + lbl[:-4] +'.png')
        plt.savefig(figpath)

    
def checkDV(float_coords, root_vols):
    #check if a neuron is dorsal or ventral; return True if dorsal
    #array input is sorted by z coordinate where last value is most rostral
    if float_coords[-1][2]<float_coords[0][2]:
        #if the last value in the list isn't the most rostral
        print('Coordinates are in the wrong order in checkDV')
        
    nm = ['ventral', 'dorsal']
    #check the neuron - is the midpoint in the dorsal or ventral roots? 
    for j in range(2):
        #check midpoint
        dst = model.GetEntityPointDistance(root_vols[j], Vec3(
            float_coords[int(len(float_coords)/2)])).Distance
        if dst <= 0:  # inside a given volume
            #print(nm[j])
            #also check both ends. 
            d1 = model.GetEntityPointDistance(root_vols[j], Vec3(
                float_coords[0])).Distance
            d2 = model.GetEntityPointDistance(root_vols[j], Vec3(
                float_coords[-1])).Distance
            if d2>0: # if rostral end not in same volume, check gray and whitematter
                gd = model.GetEntityPointDistance(root_vols[2], Vec3(
                    float_coords[-1])).Distance
                wd = model.GetEntityPointDistance(root_vols[3], Vec3(
                    float_coords[-1])).Distance
                if gd>0 and wd>0:
                    break
            if d1<=0:
                return not bool(j)
            else:
                break

    print('not assigned')   
    return -1
    #find the most rostral z coordinate
    #if at the rostral end y>.7, dorsal
    # return float_coords[-1][1]>cutoff

def checkLR(float_coords, cutoff, filename=''):
    #check if a neuron is left or right; return string "Left" or "Right"
    #Left is animal left, Right is animal's right
    #array input is sorted by z coord where first value is most caudal 
    # (and typically is furthest from midline to avoid confusing classification)
    #check these all!! TODO
    if float_coords[-1][2]<float_coords[0][2]:
        #if the last value in the list isn't the most rostral
        print('Coordinates are in the wrong order in checkLR')
    
    if '_r' in filename:
        return ' Right'
    elif '_l' in filename:
        return ' Left'

    if float_coords[0][0]>cutoff:
        #for this one check the most caudal point in the neuron
        return ' Left'
    else:
        return ' Right'    

def makeNeuron(locArray, nname):
    #location array is 3 x n array of points to define the spline
    vecList = []
    if len(locArray)<2:
        return

    for i in range(len(locArray)):
        #if locArray[i][2]>-10 and locArray[i][2]<15:
        vecList.append(Vec3(locArray[i]))

    if len(vecList)<2:
        print('Short neuron is out of bounds')
        return

    axon = model.CreateSpline(vecList)
    axon.Name = nname
    return axon

def str2coords_convert(input_str):
    #convert string to list of floats
    templist = input_str.split(' ')
    if not templist[-1]:
        res = [float(idx) for idx in input_str.split(' ')[:-1]]
    else: #no empty string at end
        res = [float(idx) for idx in input_str.split(' ')]
    #now convert list of floats to coordinates in format [[xn, yn, zn], ...]
    float_coords = [res[x:x+3] for x in range(0, len(res), 3)]

    #TODO: convert these into coordinates appropriate for the actual values we're in (roughly 3.8-38) and rotate
    #TODO: these are temp coord conversions until we get voxels sorted
    scalefactor = 10 #5.4 #should be a little longer than cord - 200 um mapped onto 1 mm; 5x conversion
    #scaleoffset = [-9, 12.4, 43] #original/simplified model
    # scaleoffset = [-38.5, -9.5, -20] #complex model cat 1
    #scaleoffset = [-12.7, 20.1, -12.2]  # complex model cat 3
    scaleoffset = [0, 0, 0]
    
    for i in range(len(float_coords)):
        if len(float_coords[i])<3:
            continue
        #this is for the simple model
        # float_coords[i] = [float_coords[i][1]/scalefactor+scaleoffset[0], -float_coords[i]
        #                    [0]/scalefactor+scaleoffset[1], -float_coords[i][2]/scalefactor + scaleoffset[2]]
        float_coords[i] = [float_coords[i][0]/scalefactor+scaleoffset[0], float_coords[i]
                           [1]/scalefactor+scaleoffset[1], float_coords[i][2]/scalefactor + scaleoffset[2]]
    #for complex cat 3, ignore any coord with z larger than 76 mm (clip off from the rostral end)
    return [x for x in float_coords if x[2]<76]


def str2coords(input_str):
    #convert string to list of floats
    templist = input_str.split(' ')
    if not templist[-1]:
        res = [float(idx) for idx in input_str.split(' ')[:-1]]
    else:  # no empty string at end
        res = [float(idx) for idx in input_str.split(' ')]
    #now convert list of floats to coordinates in format [[xn, yn, zn], ...]
    float_coords = [res[x:x+3] for x in range(0, len(res), 3)]

    return float_coords

def coord2str(input_lst):
    #convert list of floats to a string in format for rewriting coords to file
    #converts a 2D list, so a single list of coords
    return ' '.join([' '.join(['{:.3f}'.format(x) for x in l]) for l in input_lst])
