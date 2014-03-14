#!/usr/bin/env python

import numpy as np
from matplotlib import pyplot as plt

def vdot(x, y):
    #return np.sum(np.multiply(x, y), axis=-1)
    return np.einsum("...i,...i->...", x, y)

def vangle(x,y):
    return np.arccos(vdot(x,y)/vlen(x)/vlen(y))*180/np.pi

def vlen(x):
    return np.sqrt(vdot(x,x))
    
def vnorm(x):
    return np.asarray(x)/vlen(x)[...,np.newaxis]

def allv(max=10):
    return np.mgrid[-max:max+1, -max:max+1, -max:max+1].reshape(3,-1).T

# http://physics.nist.gov/cgi-bin/cuu/Value?mec2mev
m0 = 0.510998910e6 # eV/c^2
# http://physics.nist.gov/cgi-bin/cuu/Value?hbcmevf
hbar = 197.3269 # MeV.fm/c = eV.nm/c

def klen(U):
    """length of kvector of relativistic electron"""
    return np.sqrt(U*(U + 2*m0))/hbar




def equal_aspect(size=None):
    if size is not None:
       plt.xlim(-size, size)   
       plt.ylim(-size, size)   
    plt.gca().set_aspect("equal")
    #plt.axis("equal")

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
    phi = np.asarray(phi)/180.*np.pi
    r = np.zeros(phi.shape + (3,3))
    a,b,c = axes
    r[...,a,a] = r[...,b,b] = np.cos(phi)
    r[...,c,c] = 1.
    r[...,b,a] = np.sin(phi)
    r[...,a,b] = -r[...,b,a]
    return r

def circle(v, b=0, npoints=500):
    from numpy import sin, cos, pi, linspace, transpose

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



