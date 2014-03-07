#!/usr/bin/env python

from util import *
from projection import Stereo, ThreeD
from holder import DoubleTilt, TiltRotation
import structure
import format 

def wulf(minor=2, major=10, ax=None):
    """draws Wulf's net"""
    DoubleTilt().grid(minor, major, ax=ax, proj=stereo)
