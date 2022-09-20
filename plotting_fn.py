'''plotting functions for analysis'''
import os
from statistics import median
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import random
from helper_fn import *

def generateNDist(data, inc_reclocs, perc_neurons):
    '''INPUTS: data, as a pandas data structure that contains the columns 
    ['name', 'nerve', 'diameter', 'elec_dist', 'enc_tissue', 'rec_amp'], 
    where name includes root level, 'Left' or 'Right' and 'Dorsal' or 'Ventral' to ID
    the neurons. inc_reclocs, as a list of strings indicating location of 
    first recruitment that should be included in final output.
    OUTPUT: same data structure, but with only the rows matching the given
    recruitment tissue, subselected and assigned to nerves by size distribution, and
    with specific columns labeling root, side, and type of neuron'''
    #remove data that is recruited in other tissues
    print('Generate neuron distribution')
    tmp = data[data['enc_tissue'].isin(inc_reclocs)].copy()
    #NEURON DISTRIBUTIONS
    #Axons per nerve - ratio at each level (each level sums to 100); [pelvic, pudendal, sciatic] %
    v_ratio = {'L6': [0, 0, 100], 'L7': [0, 0, 100], 'S1': [
        7, 62, 31], 'S2': [66, 29, 5], 'S3': [13, 87, 0]}
    d_ratio = {'L6': [0, 0, 100], 'L7': [0, 0, 100], 'S1': [
        9, 22, 69], 'S2': [55, 37, 8], 'S3': [64, 36, 0]}
    root_names = list(v_ratio.keys())
    #Typical size distribution for each NERVE - in same order as the v_ratio and d_ratio arrays (pelvic, pudendal, sciatic)
    sizebins = range(1, 21)
    d_sizedist = [[0, 55.9, 26.4, 5.9, 5.9, 5.9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0.6, 3.5, 7.3, 10.3, 12.8, 13.8, 13.1, 12.1, 10.2,
                    8, 4.8, 1.2, 0.9, 0.7, 0.3, 0.4, 0, 0, 0, 0],
                [0, 0, 0.2, 4.5, 8.9, 10.0, 7.9, 6.9, 4.9, 4.5, 5.2, 5.3, 5.3, 6.7, 7.4, 7.3, 6.6, 4.8, 2.5, 1.1]]
    v_sizedist = [[0, 41.7, 50.9, 7.4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 4.6, 4.9, 11.5, 4.6, 5.9, 17.5, 13.2,
                    9, 6.2, 8.2, 0.1, 4.4, 0.9, 4.6, 4.4, 0, 0],
                [0, 0, 0.2, 4.5, 8.9, 10.0, 7.9, 6.9, 4.9, 4.5, 5.2, 5.3, 5.3, 6.7, 7.4, 7.3, 6.6, 4.8, 2.5, 1.1]]

    #add labels based on name of each neuron
    spinal_lvl_idx = np.empty(len(tmp), dtype='<U2')

    for i in range(len(root_names)):
        spinal_lvl_idx[tmp['name'].str.contains(root_names[i])] = root_names[i]

    tmp['root'] = spinal_lvl_idx
    tmp['type'] = [['afferent', 'efferent'][i]
                   for i in tmp['name'].str.lower().str.contains('ventral')*1]
    tmp['side'] = [['right', 'left'][i] for i in tmp['name'].str.lower().str.contains('left')*1]

    #based on these numbers and the neuron distributions, how many axons should be assigned to each nerve?
    #count the number of ie dorsal L6 neurons, multiply it by distribution in ratios
    d_arr = [[] for x in root_names]
    v_arr = [[] for x in root_names]
    nerve_idx = [-1 for x in range(len(tmp))]
    tmp['nerve'] = nerve_idx
    for i in range(len(root_names)):
        #assign how many of each nerve, based on level - do 50% of total neurons
        aff_tmp = tmp[(tmp['root'] == root_names[i]) & (tmp['type'] == 'afferent')]
        eff_tmp = tmp[(tmp['root'] == root_names[i]) & (tmp['type'] == 'efferent')]
        ddist = [x*.0001*perc_neurons*len(aff_tmp) for x in d_ratio[root_names[i]]]
        vdist = [x*.0001*perc_neurons*len(eff_tmp) for x in v_ratio[root_names[i]]]

        #the only reason to index this like this is to be able to save the whole
        # expected distribution to compare
        d_arr[i] = [[] for x in range(len(ddist))]
        v_arr[i] = [[] for x in range(len(ddist))]
        for j in range(len(ddist)):
            d_arr[i][j] = np.random.choice(sizebins, int(ddist[j]), p=[
                x*.01 for x in d_sizedist[j]])
            v_arr[i][j] = np.random.choice(sizebins, int(vdist[j]), p=[
                x*.01 for x in v_sizedist[j]])

        used_idx = []
        missing_n = [[] for x in range(len(ddist))]
        #find diameters of neurons that are AFFERENT and in the appropriate root
        for j in range(len(ddist)):  # for each nerve
            # for each size that was randomly called for in dorsal dist
            for k in range(len(d_arr[i][j])):
                #find an appropriate neuron.
                if d_arr[i][j][k] in list(aff_tmp['diameter']):
                    #find indices that have not already been chosen
                    remain_idx = list(
                        set(aff_tmp.index[aff_tmp['diameter'] == d_arr[i][j][k]])-set(used_idx))
                    if len(remain_idx) == 0:
                        missing_n[j].append(d_arr[i][j][k])
                        print('No remaining neuron of size ' + str(d_arr[i][j][k]))
                        #for now, just print this and then don't do anything about it... need to figure out how common it is
                        continue
                    #pick a random index
                    n_idx = random.choice(remain_idx)
                    #add it to the list of chosen indices - should be fastest to just save idx in separate list
                    tmp['nerve'].loc[n_idx] = j
                    used_idx.append(n_idx)
                else:
                    missing_n[j].append(d_arr[i][j][k])
                    print('No neuron of size ' + str(d_arr[i][j][k]))

        #Do the same for neurons that are EFFERENT
        for j in range(len(vdist)):  # for each nerve
            # for each size that was randomly called for in vebtral dist
            for k in range(len(v_arr[i][j])):
                #find an appropriate neuron.
                if v_arr[i][j][k] in list(eff_tmp['diameter']):
                    #find indices that have not already been chosen
                    remain_idx = list(
                        set(eff_tmp.index[eff_tmp['diameter'] == v_arr[i][j][k]])-set(used_idx))
                    if len(remain_idx) == 0:
                        missing_n[j].append(v_arr[i][j][k])
                        print('No remaining neuron of size ' +
                              str(v_arr[i][j][k]))
                        #for now, just print this and then don't do anything about it... need to figure out how common it is
                        continue
                    #pick a random index
                    n_idx = random.choice(remain_idx)
                    #update list info 
                    tmp['nerve'].loc[n_idx] = j
                    used_idx.append(n_idx)
                else:
                    missing_n[j].append(v_arr[i][j][k])
                    print('No neuron of size ' + str(v_arr[i][j][k]))

    tmp['nerve'] = [['pelvic', 'pudendal', 'sciatic', 'undefined'][i] for i in tmp['nerve']]
    #TODO also output: d_sizedist, v_sizedist, d_arr, v_arr, sizebins
    expected_sizedist = {'sizes':sizebins, 'dorsal':d_sizedist, 'ventral':v_sizedist}
    actual_sizelst = {'dorsal':d_arr, 'ventral':v_arr}
    return tmp[tmp['nerve'] != 'undefined'], expected_sizedist, actual_sizelst




def plotNDist(ideal_dist, actual_dist, bins, savepath=''):
    '''input the ideal distribution as percent values for each of the bins, 
    input the actual distribution as the list of values assigned'''
    if savepath!='':
        f = open(savepath, 'w')
        lbl = savepath.split('neuronDist')[1]
    #print('this is new' + '2')
    allkeys = [x for x in ideal_dist.keys()]
    #print(actual_dist)
    binmid = [(x + y)/2+(y-x) for (x, y) in zip(bins[:-1], bins[1:])]
    for key in allkeys:
        # print(actual_dist[key])
        # print(ideal_dist[key])
        if len(actual_dist[key])==0:
            # print(key)
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
        # print(vals)
        # print(len(vals))
        # print(binmid)
        # print(len(binmid))
        plt.scatter(binmid, vals, marker = 'o', alpha = 0.8, c=[[0, 0, 0]])

        box_off(ax)
        imax = max([max(actual_dist[key]), max(vals)])
        plt.ylim([0, imax*1.2])
        reset_ticks(ax, [0, bins[-1]], [0, imax])

        if savepath!='':
            #save to file
            f.write(key + '\n')
            f.write('bins ' + str(binmid) + '\n')
            f.write('actual ' + str(actual_dist[key]) + '\n')
            f.write('ideal ' + str(vals) + '\n')

            figpath = os.path.join(savepath, '..', key.lower() + '_dist' + lbl[:-4] +'.png')
            plt.savefig(figpath)

def plotRootRec(df, figpath=''):
    '''input a filepath to save and a pandas dataframe containing afferents/efferents labeled
    in the "type" column and roots labeled in "root" column'''
    tmp1 = pd.melt(df, id_vars=['type'], value_vars=['root'])

    tmp2 = pd.melt(df, id_vars=['type'], value_vars=[
                'rec_amp'], value_name='Rec Amp', var_name='Root')
    tmp2['Root'] = tmp1['value']  # just have to massage this so labels are correct
    tmp2.sort_values(by='type') #this should keep the ordering same in all plots

    clr = ["#ffa800", "#a560bd"]
    newPal = dict(afferent=clr[1], efferent=clr[0])
    newPalDk = dict(afferent="#45005e", efferent="#a53d00")

    # Initialize the figure
    f, ax = plt.subplots()
    sns.despine(left=True)

    # Show each observation with a scatterplot
    sns.stripplot(x="Rec Amp", y="Root", hue="type",
                data=tmp2, dodge=True, alpha=.25, zorder=1, palette=newPal)

    # Show the conditional means, aligning each pointplot in the
    # center of the strips by adjusting the width allotted to each
    # category (.8 by default) by the number of hue levels
    sns.pointplot(x="Rec Amp", y="Root", hue="type",
                data=tmp2, dodge=.8 - .8 / 2,
                join=False, palette=newPalDk,
                markers="d", scale=.75, ci=None, estimator=median)

    # Improve the legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[2:], labels[2:], title="Type",
            handletextpad=0, columnspacing=1,
            loc="lower right", ncol=2, frameon=True)
    if figpath:
        plt.savefig(figpath)
