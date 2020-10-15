
from .util import *
from .projection import Stereo, ThreeD, Laue, Flat, Cut, Projection, Persp
from .holder import DoubleTilt, TiltRotation, DoubleTilt2
from .structure import Structure
from . import structure
from . import format
from . import cif

def wulf(minor=2, major=10, ax=None):
    """draws Wulf's net"""
    DoubleTilt().grid(minor, major, ax=ax, proj=Stereo())
