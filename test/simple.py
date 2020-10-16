#!/usr/bin/python3

from empy import *

s = structure.bcc(0.29)
o = s.orient([1,1,1])
p = Stereo()
v = allv(1)

p.points(o(v), c="r")
p.labels(o(v), map(format.tex, v))
p.circles(o(equiv([0,0,1],[1,1,0])), c="r")

plt.show()

