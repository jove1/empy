#!/usr/bin/env python 

from setuptools import setup

import subprocess
try:
    a,b,c = subprocess.check_output(["git", "describe", "--tags", "--always"]).strip("v\n").split("-")
    __version__ = "{}.{}".format(a,b)
except subprocess.CalledProcessError:
    execfile('empy/version.py') # __version__
else:
    with open('empy/version.py','w') as f:
        f.write("__version__ = {!r}\n".format(__version__))

setup(
    name='empy',
    version = __version__,
    description='Python electron microscopy tools.',
    author = 'Jozef Vesely',
    author_email = 'vesely@gjh.sk',
    url = 'http://github.com/jove1/empy',
    
    packages = ["empy"],
    install_requires = ['numpy', 'matplotlib'],
)
