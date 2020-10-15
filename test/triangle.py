#!/usr/bin/python3

from empy import *

a, b = np.meshgrid(linspace(0,1), linspace(0,1))
v = vnorm(np.dstack([a, a*b, np.ones_like(a)]))
c = vrgb(v)
v = v[...,:2]/(1+v[...,2])[...,newaxis]

from matplotlib.collections import QuadMesh
q = QuadMesh(
        v.shape[0]-1, v.shape[1]-1, 
        v.reshape(-1,2), 
        color=c.reshape(-1,3),
        zorder=0,
        shading='gouraud'
        )

p = Stereo(border='std', ax=plt.figure(figsize=(4.4,4)).add_axes([0,0,1,1]))
p.ax.add_collection(q)
o = Orient.std()
v = [(0,0,1),(0,1,1),(-1,1,1)]
p.labels(o(v), map(format.tex,v), va='center')
plt.savefig("colormap.pdf")
plt.show()

