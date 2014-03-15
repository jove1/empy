#!/usr/bin/env python

from .util import *
from .projection import Stereo

class Holder:
    def get_proj(self, d):
        try:
            return d['proj']
        except KeyError:
            return Stereo()

    def __call__(self, *args):
        return self.rot(*args)[...,2,:] # inverse of rotation matrix -> same as transpose

    npoints = 500
    def grid(self, minor=10, major=None, **kwargs):
        proj = self.get_proj(kwargs)

        for a in range(-180, 181, minor):
            b = np.linspace(-180, 180, self.npoints)

            lw = 0.5
            if major and a % major == 0:
                lw = 1.2

            proj.plot(self(a, b), "k-", lw=lw, zorder=0)
            proj.plot(self(b, a), "k-", lw=lw, zorder=0)

class DoubleTilt(Holder):
    def rot(self, phi, theta): 
        R1 = axis_rot(np.asarray(phi), (2,1,0))
        R2 = axis_rot(np.asarray(theta), (0,2,1))
        return np.einsum("...ij,...jk->...ik", R2, R1)
    
    maxphi = 40
    maxtheta = 30
    def range(self, **kwargs):
        proj = self.get_proj(kwargs)

        maxphi = self.maxphi
        maxtheta = self.maxtheta
        theta = np.linspace(-maxtheta, maxtheta, self.npoints)
        phi = np.linspace(-maxphi, maxphi, self.npoints)
        proj.plot(self(maxphi, theta), "k-", lw=2)
        proj.plot(self(-maxphi, theta), "k-", lw=2)
        proj.plot(self(phi, maxtheta), "k-", lw=2)
        proj.plot(self(phi, -maxtheta), "k-", lw=2)

        theta = np.linspace(0, maxtheta, self.npoints/2)
        phi = np.linspace(0, maxphi, self.npoints/2)
 
        proj.plot(self(0, theta), "k--", lw=2)
        proj.plot(self(0, -theta), "k--", lw=2)
        proj.plot(self(phi, 0), "k--", lw=2)
        proj.plot(self(-phi, 0), "k--", lw=2)
        
class TiltRotation(Holder):
    def rot(self, phi, theta): 
        R1 = axis_rot(np.asarray(phi), (1,0,2))
        R2 = axis_rot(np.asarray(theta), (0,2,1))
        return np.einsum("...ij,...jk->...ik", R2, R1)

    maxtheta = 30
    def range(self, **kwargs):
        proj = self.get_proj(kwargs)

        maxtheta = self.maxtheta
        theta = np.linspace(0, maxtheta, self.npoints)
        phi = np.linspace(-180, 180, self.npoints)
        proj.plot(self(phi, maxtheta), "k-", lw=2)

        proj.plot(self(0, theta), "k-", lw=2)
        proj.plot(self(90, theta), "k--", lw=2)
        proj.plot(self(180, theta), "k--", lw=2)
        proj.plot(self(270, theta), "k--", lw=2)
