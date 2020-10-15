#!/usr/bin/python3

from empy import *
phase = structure.fcc(0.4)
v = maxhkl(6, True)
v = v[phase.hkl_allowed(v)]


for l, o in [
    ("100", phase.orient([1,0,0], [0,0,1])),
    ("110", phase.orient([1,1,0], [0,0,1])),
    ("111", phase.orient([1,1,1], [0,-1,1])),
    ("112", phase.orient([2,1,1], [0,-1,1])),
    ]:
    q = o(phase.q(v))
    p = Persp(ax=a4figure("landscape").gca())
    plt.title(l)
    p.kikuchi(q)
    p.equal_aspect(0.25)


o = phase.orient([2.5,1,1], [0,1,-1], -45)
q = o(phase.q(v))
p = Stereo(ax=a4figure("landscape").gca(), border=False)
p.kikuchi(q)
p.equal_aspect(0.25)


plt.show()
