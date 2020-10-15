#!/usr/bin/python3

from empy import *

s = structure.bcc(0.28)
o = [1,0,0]
p = Cut()
v = maxhkl(6, True)

plt.title(str(o))
oo = s.orient(o)
v = v[s.hkl_allowed(v)]
vv = oo(s.q(v))
p.points(vv, s='auto')
p.labels(vv, map(format.simple, v))
plt.savefig("simple2.pdf")
plt.show()
