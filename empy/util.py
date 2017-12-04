#!/usr/bin/env python

import numpy as np
from numpy import pi, cos, sin, deg2rad, rad2deg, cross, exp, sqrt, array, newaxis, linspace, dot, nan, log, abs, hypot

from matplotlib import pyplot as plt

def vdot(x, y):
    #return np.sum(np.multiply(x, y), axis=-1) # einsum is faster
    return np.einsum("...i,...i->...", x, y)

def vangle(x,y):
    #return rad2deg(np.arccos(vdot(x,y)/vlen(x)/vlen(y)))
    # this avoids normalisation problems, and is more stable near 0deg
    return rad2deg(np.arctan2(vlen(cross(x,y)), vdot(x,y)))

def vlen(x):
    return np.sqrt(vdot(x,x))
    
def vnorm(x):
    return np.asarray(x)/vlen(x)[...,np.newaxis]

def allv(max=10):
    return np.mgrid[-max:max+1, -max:max+1, -max:max+1].reshape(3,-1).T

def maxhkl(maxhkl, sphere=False):
    v = allv(maxhkl)
    if sphere:
        return v[vlen(v)<=maxhkl]
    return v

def sym_expand(v, gen, prec=7):
    l = [v]
    seen = set()
    for v in l:
        u = round(v[0] % 1, prec), round(v[1] % 1, prec), round(v[2] % 1, prec)
        if u in seen:
            continue
        seen.add(u)
        l.extend(gen(*v))
    return list(seen)

def equiv(*hkls):
    permutations = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
    signs = [(-1, -1, -1), (-1, -1, 1), (-1, 1, -1), (-1, 1, 1),
             (1, -1, -1),  (1, -1, 1),  (1, 1, -1),  (1, 1, 1)]
    return list(set( (sgn[0]*hkl[idx[0]], sgn[1]*hkl[idx[1]], sgn[2]*hkl[idx[2]])
                for idx in permutations 
                for sgn in signs 
                for hkl in hkls))

def normhkl(hkl):
    return np.sort(abs(hkl), axis=-1)

def normhkl_hex(hkl):
    l = hkl[...,2] # save l
    hkl = hkl.copy()
    hkl[...,2] = -(hkl[...,0] + hkl[...,1]) # compute 3rd index
    hkl = np.sort(abs(hkl), axis=-1) # biggest is last (should be with minus sign)
    hkl[...,2] = abs(l)  # restore l  
    return hkl

def equiv_hex(*hkls):
    signs = [(-1,-1), (-1,1), (1,-1), (1,1)]
    return list(set( (sgn[0]*hk[0], sgn[0]*hk[1], sgn[1]*hkl[2])
                for hkl in hkls
                for hk in [ (hkl[0], hkl[1]),
                            (hkl[1], hkl[0]),
                            (hkl[0], -hkl[0]-hkl[1]),
                            (hkl[1], -hkl[0]-hkl[1]),
                            (-hkl[0]-hkl[1], hkl[0]),
                            (-hkl[0]-hkl[1], hkl[1])]
                for sgn in signs))

# http://physics.nist.gov/cgi-bin/cuu/Value?mec2mev
m0 = 0.510998910e6 # eV/c^2
# http://physics.nist.gov/cgi-bin/cuu/Value?hbcmevf
hbar = 197.3269 # MeV.fm/c = eV.nm/c

def klen(U):
    """length of kvector of relativistic electron"""
    return np.sqrt(U*(U + 2*m0))/hbar

def ex_error(g, k=[0,0,klen(200e3)], n=[0,0,1]):
    n = np.asarray(n)
    k = np.asarray(k)
    # (k + g + e*n)**2 = k**2
    # (k+g)**2 - k**2 + e 2n*(k+g) + e**2 = 0
    #
    # x**2 + 2bx + c = 0
    b = dot(k+g, n)/vlen(n)
    # (k+g)**2 - k**2 = 2*k*g + g**2
    c = vdot(2*k+g, g) 
    
    # exact solution avoiding subtraction of close values
    # 1 + 2b/x + c/x**2 = 0 
    # 1/x = (-b +- sqrt(b*b-c))/c
    # x = c/(-b +- sqrt(b*b-c))
    return np.where( b>=0, c/(-b-sqrt(b*b-c)), c/(-b+sqrt(b*b-c)))


def sinc(x):
    return np.where(x!=0, sin(x)/x, 1)

def a4figure(orient="portrait"):
    if orient == "portrait":
        return plt.figure(figsize=(8.27, 11.69), dpi=72)
    elif orient == "square":
        return plt.figure(figsize=(8.27, 8.27), dpi=72)
    else:
        return plt.figure(figsize=(11.69, 8.27), dpi=72)

def tight_borders():
    plt.subplots_adjust(0,0,1,1,0,0)


def fit_zone(v):
    U,s,V = np.linalg.svd(v)
    return V[-1]

def flip(v):
    v = np.array(v)
    m = v[...,2]<0
    v[m] = -v[m]
    return v

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

    a = linspace(-pi, pi, npoints)
    #a = linspace(0, pi, npoints)
    x = sin(a)*cos(b) 
    y = cos(a)*cos(b)
    z = sin(b)

    x,z = rot(ct, st, x, z)    
    x,y = rot(-cp, -sp, x, y)

    return transpose([x,y,z])

def quat2rot(v):
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
    # http://en.wikipedia.org/wiki/Kabsch_algorithm
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
    return quat2rot( v[:,np.argmax(w)] ) 


def vrgb(v):
    xyz = np.sort(abs(v), axis=-1)
    rgb = np.empty(xyz.shape)
    rgb[...,0] = xyz[...,2]-xyz[...,1]
    rgb[...,1] = (xyz[...,1]-xyz[...,0])*np.sqrt(2)
    rgb[...,2] = xyz[...,0]*np.sqrt(3)

    # due to this we don't have to take vnorm at the beginning
    rgb /= np.amax(rgb, axis=-1)[...,np.newaxis] 
    return rgb 

def kwalias(d, a, b):
    try: d[a] = d.pop(b)
    except KeyError: pass

class Orient:
    def __init__(self, rotmatrix=None):
        if rotmatrix is None:
            self.rotmatrix = np.diag([1.,1.,1.])
        else:
            self.rotmatrix = rotmatrix

    def __call__(self, vec):
        if isinstance(vec, Orient):
            return Orient(np.dot(vec.rotmatrix, self.rotmatrix))
        return np.dot(vec, self.rotmatrix)

    def __repr__(self):
        return "Orient({!r})".format(self.rotmatrix)

    def inv(self):
        return Orient(self.rotmatrix.T)
    
    def normal(self):
        return self.rotmatrix[:,2]
    
    def perp(self, a):
        return np.dot(axis_rot(a)[...,0], self.rotmatrix.T)

    @staticmethod
    def fit(lst):
        return Orient(fit_rot(lst))

    @staticmethod
    def std():
        return Orient([[0,-1,0],[1,0,0],[0,0,1]])

import format as _format
def orient_plot(p, o, c="k", max=2, format=_format.simple, ls="-"):
    v = equiv([0,0,1],[0,1,1])
    p.circles(o(v), c=c, ls=ls)

    v = allv(max)
    p.points(o(v), c=c)
    p.labels(o(v), map(format, v), color=c)


class Container(dict):
    def __setattr__(self, k, v):
        self[k] = v

    def __getattr__(self, k):
        return self[k]

    def write(self, fh):
        for k,v in self.items():
            fh.write("{} = {!r}\n".format(k,v))

def load_conf(fname):
    import os
    d = Container()
    path = os.path.dirname(fname)
    fname = os.path.basename(fname)
    wd = os.getcwd()
    if path:
        os.chdir(path)

    try:
        execfile(fname, d)
    finally:
        os.chdir(wd)

    del d['__builtins__']

    return d

def load_kik_file(arg):
    if isinstance(arg, Container):
        conf = arg
    else:
        conf = load_conf(arg)

    s = conf.get("s")
    if s is None:
        import Structure
        s = Structure.bcc(0.29)
        import warnings
        warnings.warn("Assuming cubic crystal in load_kik_file")

    def tr(x):
        if len(x) == 2:
            phi, theta = map(deg2rad, x)
            return (cos(phi)*sin(theta), sin(phi)*sin(theta), cos(theta))
        else:
            return x
    return Orient.fit([ (tr(a), s.q(b)) for a,b in conf.measurements if None not in (a,b)])

    #data = [ (tr(a), b) for a,b in conf.measurements if None not in (a,b)]
    #r = Orient.fit(data)
    #print len(data), np.mean([ vangle(a,r(b)) for a,b in data])
    #return r


def load_laue_file(fname):
    spots = []
    dirs = []
    for l in open(fname):
        l = l.split()
        if l[0] == "distance":
            d = float(l[1])

        elif l[0] == "dir":
            ll = map(float, l[1:])
            if len(ll) == 5:
                dirs.append(ll)

        elif l[0] == "spot":
            ll = map(float, l[1:])
            spots.append(ll)
    
    if not dirs:
        raise ValueError, ("No indexing info in laue file.", fname)
    
    import warnings
    warnings.warn("Assuming cubic crystal in load_laue_file")
    orient = Orient.fit([ ((2.*x/(1+x*x+y*y),
                            2.*y/(1+x*x+y*y),
                            2./(1+x*x+y*y)-1), (h,k,l) ) for x,y, h,k,l in dirs ])
    spots = np.array([ (x,
                        y,
                        np.sqrt(x*x + y*y + d*d)+d) for x,y in spots])
    return orient, spots
