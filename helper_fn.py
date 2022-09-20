'''Helper classes and functions to run Sim4Life modeling'''
import numpy

#basically just need to define these classes because python 3.6 and below don't have mutable structs
class Material(object):
    ''' Defines the Material settings for an Unstructured LF simulation '''
    def __init__(self,name,domains,type,typediel,sigma):
        self.name=name
        self.list=domains
        self.type=type
        self.typediel=typediel
        self.sigma=sigma
        return

class Boundaries(object):
    ''' Defines the Boudary settings for an Unstructured LF simulation '''
    def __init__(self,patch,type,value):
        self.entity=patch
        self.name=patch.Name
        self.type=type
        self.value=value
        return    

class UnstructuredModel(object):
    ''' Defines the settings for an Unstructured Model '''
    def __init__(self,Name):
        self.materialslist=[]
        self.boundarieslist=[]
        self.thinlayerslist=[] #TODO not sure if this part is necessary? from electroceuticals tutorial
        return    

class ModelComponent():
    def __init__(self, name, dielectric, localmeshsize, priority):
        self.name = name
        self.dielectric = dielectric
        self.localmesh = localmeshsize
        self.priority = priority

class Fiber():
    def __init__(self, name_str, diameter_um):
        self.name = name_str
        self.diameter = diameter_um

class Axon(): 
    def __init__(self, neuron_lbl, afferent_bool, nerve_idx, diameter_um, class_str, spinal_lvl):
        '''afferent_bool is a true/false indicating if the axon is afferent/efferent
        nerve_idx is 1 for pelvic, 2 for pudendal, 3 for sciatic, 0 for other (ie that we don't care about)
        diameter in microns
        string describing neuron class to assign (unmyelinated/C, A-delta, large/Ab and Aa)
        string describing spinal level '''
        self.name = neuron_lbl
        self.dorsal = afferent_bool
        self.nerve = nerve_idx
        self.diameter = diameter_um
        self.type = class_str
        self.lvl = spinal_lvl

    def nerve_id(self):
        nerve_dict = {1:'Pelvic', 2:'Pudendal', 3:'Sciatic', 0:'Other'}
        return nerve_dict[self.nerve]

#These fxns are helper functions
def GetDomainsMesh(mesh):
    domains = {}
    dom = mesh.Domains
    for d in dom:
        domains[d.Name]=d        
    return domains


def find_nearest(val, alist): 
    '''Finds index of nearest neighbor in a 1D list'''
    near_curve = [abs(x-val) for x in alist]
    return near_curve.index(min(near_curve))

def find_nearest2D(val, alist):
    '''Finds index of nearest neighbor given an x,y pair and a list of coords'''
    near_curve = [numpy.sqrt((val[0]-x[0])**2 + (val[1]-x[1])**2) for x in alist]
    return near_curve.index(min(near_curve))

def find_nearest3D(val, alist):
    '''Finds index of nearest neighbor given an x,y,z pair and a list of coords'''
    near_curve = [numpy.sqrt((val[0]-x[0])**2 + (val[1]-x[1])**2 + (val[2]-x[2])**2) for x in alist]
    return near_curve.index(min(near_curve))

def list_idx(alist, idxlist):
    '''Returns the values in a list that are located at all indices in idxlist. 
    Deals with "can't index with list" problem'''
    return [x for x in map(alist.__getitem__, idxlist)]

def box_off(ax):
    '''Removes the outline of the plot on the upper and right sides'''
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def reset_ticks(ax, data_xlim, data_ylim):
    '''Sets ticks to be at the data min/max (so the plot specifically marks the full
    range and has a tick at the maxes)'''
    xticks = ax.get_xticks()
    for l in range(len(xticks)):
        if xticks[l]>data_xlim[0]:
            break
    for r in range(len(xticks)-1, 0, -1):
        if xticks[r]<data_xlim[1]:
            break
    ax.set_xlim(xticks[l-1], xticks[r+1])

    yticks = ax.get_yticks()
    for u in range(len(yticks)):
        if yticks[u]>data_ylim[0]:
            break
    for d in range(len(yticks)-1, 0, -1):
        if yticks[d]<data_ylim[1]:
            break
    ax.set_ylim(yticks[u-1], yticks[d+1])


def set_corner(ax, minx, miny):
    ax.spines['left'].set_position(('data', minx))
    ax.spines['bottom'].set_position(('data', miny))
