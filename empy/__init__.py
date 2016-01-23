#!/usr/bin/env python

from util import *
from projection import Stereo, ThreeD, Laue, Flat, Cut, Projection, Persp
from holder import DoubleTilt, TiltRotation, DoubleTilt2
import structure
import format 

def wulf(minor=2, major=10, ax=None):
    """draws Wulf's net"""
    DoubleTilt().grid(minor, major, ax=ax, proj=Stereo())
