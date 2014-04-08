#!/usr/bin/env python

import numpy as np
from numpy import pi, cos, sin, deg2rad, rad2deg

from matplotlib import pyplot as plt

def vdot(x, y):
    #return np.sum(np.multiply(x, y), axis=-1)
    return np.einsum("...i,...i->...", x, y)

def vangle(x,y):
    return rad2deg(np.arccos(vdot(x,y)/vlen(x)/vlen(y)))

def vlen(x):
    return np.sqrt(vdot(x,x))
    
def vnorm(x):
    return np.asarray(x)/vlen(x)[...,np.newaxis]

def allv(max=10):
    return np.mgrid[-max:max+1, -max:max+1, -max:max+1].reshape(3,-1).T


def equiv(hkl):
    permutations = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
    signs = [(-1, -1, -1), (-1, -1, 1), (-1, 1, -1), (-1, 1, 1),
             (1, -1, -1),  (1, -1, 1),  (1, 1, -1),  (1, 1, 1)]
    return set( (sgn[0]*hkl[idx[0]], sgn[1]*hkl[idx[1]], sgn[2]*hkl[idx[2]])
                for idx in permutations for sgn in signs )


# http://physics.nist.gov/cgi-bin/cuu/Value?mec2mev
m0 = 0.510998910e6 # eV/c^2
# http://physics.nist.gov/cgi-bin/cuu/Value?hbcmevf
hbar = 197.3269 # MeV.fm/c = eV.nm/c

def klen(U):
    """length of kvector of relativistic electron"""
    return np.sqrt(U*(U + 2*m0))/hbar




def a4figure(orient="portrait"):
    if orient == "portrait":
        plt.figure(figsize=(8.27, 11.69), dpi=72)
    elif orient == "square":
        plt.figure(figsize=(8.27, 8.27), dpi=72)
    else:
        plt.figure(figsize=(11.69, 8.27), dpi=72)

def tight_borders():
    plt.subplots_adjust(0,0,1,1,0,0)





def axis_rot(phi, axes=(0,1,2)):
    phi = deg2rad(phi)
    r = np.zeros(phi.shape + (3,3))
    a,b,c = axes
    r[...,a,a] = r[...,b,b] = cos(phi)
    r[...,c,c] = 1.
    r[...,b,a] = sin(phi)
    r[...,a,b] = -r[...,b,a]
    return r

def circle(v, b=0, npoints=500):
    from numpy import linspace, transpose

    r2, r3 = vlen(v[:2]), vlen(v)
    
    cp, sp = v[0]/r2, v[1]/r2
    st, ct = r2/r3, v[2]/r3

    def rot(c,s,x,y):
        return c*x-s*y, s*x+c*y

    a = linspace(0, pi, npoints)
    x = sin(a)*cos(b) 
    y = cos(a)*cos(b)
    z = sin(b)

    x,z = rot(ct, st, x, z)    
    x,y = rot(-cp, -sp, x, y)

    return transpose([x,y,z])

def quat_to_rot(v):
    r = np.outer(v, v)
    rot = np.empty((3,3))
    rot[0,0] = -r[3,3] -r[2,2] +r[1,1] +r[0,0]
    rot[0,1] = 2*(-r[0,3] + r[1,2])
    rot[0,2] = 2*( r[1,3] + r[0,2])
    rot[1,0] = 2*( r[0,3] + r[1,2])
    rot[1,1] = -r[3,3] +r[2,2] -r[1,1] +r[0,0]
    rot[1,2] = 2*( r[2,3] - r[0,1])
    rot[2,0] = 2*( r[1,3] - r[0,2])
    rot[2,1] = 2*( r[2,3] + r[0,1])
    rot[2,2] = +r[3,3] -r[2,2] -r[1,1] +r[0,0]
    assert abs(np.linalg.det(rot)-1) < 1e-5 
    return rot

def fit_rot(l):
    """finds rotation matrix that aligns two sets of vectors
    l is a list of pairs of vectors [(v1,u1), (v2,u2) ...]"""

    a = np.zeros((3,3))
    for u,v in l:
        a += np.outer(vnorm(v),vnorm(u))
   
    b = np.empty((4,4))
    b[0,0] = a[2,2] + a[0,0] + a[1][1]
    b[0,1] = -a[1,2] + a[2,1]
    b[0,2] = a[0,2] - a[2,0]
    b[0,3] = a[1,0] - a[0,1]
    b[1,0] = -a[1,2] + a[2,1]
    b[1,1] = -a[2,2] - a[1,1] + a[0][0]
    b[1,2] = a[1,0] + a[0,1]
    b[1,3] = a[2,0] + a[0,2]
    b[2,0] = a[0,2] - a[2,0]
    b[2,1] = a[1,0] + a[0,1]
    b[2,2] = -a[0,0] - a[2,2] + a[1][1]
    b[2,3] = a[2,1] + a[1,2]
    b[3,0] = a[1,0] - a[0,1]
    b[3,1] = a[2,0] + a[0,2]
    b[3,2] = a[2,1] + a[1,2]
    b[3,3] = -a[1,1] + a[2,2] - a[0][0]

    w, v = np.linalg.eigh(b)
    i = np.argmax(w)
    return quat_to_rot( v[:,np.argmax(w)] ) 


def vrgb(v):
    xyz = np.sort(np.abs(v), axis=-1)
    rgb = np.empty(xyz.shape)
    rgb[...,0] = xyz[...,2]-xyz[...,1]
    rgb[...,1] = (xyz[...,1]-xyz[...,0])*np.sqrt(2)
    rgb[...,2] = xyz[...,0]*np.sqrt(3)

    # due to this we don't have to take vnorm at the beginning
    rgb /= np.amax(rgb, axis=-1)[...,np.newaxis] 
    return rgb 

class Orient:
    def __init__(self, rotmatrix=None):
        if rotmatrix is None:
            self.rotmatrix = np.diag([1.,1.,1.])
        else:
            self.rotmatrix = rotmatrix

    def __call__(self, vec):
        return np.dot(vec, self.rotmatrix)
    
    def normal(self):
        return self.rotmatrix[:,2]

    @staticmethod
    def fit(lst):
        return Orient(fit_rot(lst))

    @staticmethod
    def from_kik_file(fname):
        import imp
        conf = imp.load_source("conf", fname)
        def tr(x):
            phi, theta = map(deg2rad, x)
            return (cos(phi)*sin(theta), sin(phi)*sin(theta), cos(theta))
        import warnings
        warnings.warn("Assuming cubic crystal")
        return Orient.fit([ (tr(a), b) for a,b in conf.measurements ])

