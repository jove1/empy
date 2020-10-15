
from setuptools import setup

setup(
    name='empy',
    version = '0.0.1',
    description='Python electron microscopy tools.',
    author = 'Jozef Vesely',
    author_email = 'vesely@gjh.sk',
    url = 'http://github.com/jove1/empy',
    
    packages = ["empy"],
    install_requires = ['numpy', 'matplotlib'],
)
