#!/usr/bin/python3

from empy import *

p = Stereo()

s = structure.bcc(0.28)
o = s.orient([1,1,1],[-1,1,0])
c = "C1"
hklv = equiv([1,1,0],[0,0,1],[1,1,1])
hklq = equiv([0,0,1],[1,1,0])

v = o(s.v(hklv))
q = o(s.q(hklq))
p.points(v, c=c)
p.labels(v, map(format.tex,hklv), c=c)
p.circles(q, c=c)


s = structure.fcc(0.375)
o = s.orient([1,1,0],[-1,1,1])
c = "C2"
hklv = equiv([1,1,0],[0,0,1],[1,1,1])
hklq = equiv([0,0,1],[1,1,0])

v = o(s.v(hklv))
q = o(s.q(hklq))
p.points(v, c=c)
p.labels(v, map(format.tex,hklv), c=c)
p.circles(q, c=c)


p = Stereo()

s = structure.bcc(0.28)
o = s.orient([1,1,1],[-1,1,0])
c = "C1"
hklv = equiv([1,1,0],[0,0,1],[1,1,1])
hklq = equiv([0,0,1],[1,1,0])

v = o(s.v(hklv))
q = o(s.q(hklq))
p.points(v, c=c)
p.labels(v, map(format.tex,hklv), c=c)
p.circles(q, c=c)


s = structure.fcc(0.375)
o = s.orient([1,1,0],[0,0,1],60)
c = "C2"
hklv = equiv([1,1,0],[0,0,1],[1,1,1])
hklq = equiv([0,0,1],[1,1,0])

v = o(s.v(hklv))
q = o(s.q(hklq))
p.points(v, c=c)
p.labels(v, map(format.tex,hklv), c=c)
p.circles(q, c=c)

p = Stereo()

s = structure.bcc(0.33)
o = s.orient([0,0,1],[1,1,0])
c = "C1"
hklv = equiv([1,1,0],[0,0,1],[1,1,1])
hklq = equiv([0,0,1],[1,1,0])

v = o(s.v(hklv))
q = o(s.q(hklq))
p.points(v, c=c)
p.labels(v, map(format.tex,hklv), c=c)
p.circles(q, c=c)


s = structure.hcp(0.295, 0.468)
o = s.orient(hex43v([0,0,0,1]), (1,1,0))
c = "C2"

hklv = hex43v([
        [1,1,-2,0],[1,-2,1,0],[-2,1,1,0],
        [-1,-1,2,0],[-1,2,-1,0],[2,-1,-1,0],

        [1,1,-2,3],[1,-2,1,3],[-2,1,1,3],
        [-1,-1,2,3],[-1,2,-1,3],[2,-1,-1,3],

        [1,-1,0,1],[1,0,-1,1],[0,1,-1,1],
        [-1,1,0,1],[-1,0,1,1],[0,-1,1,1],

        [0,0,0,1],
        ])

hklq = [[1,-1,0],[1,0,0],[0,1,0],[-1,1,0],[-1,0,0],[0,-1,0],
        [1,-1,1],[1,0,1],[0,1,1],[-1,1,1],[-1,0,1],[0,-1,1],
        [0,0,1],[0,0,-1]]

v = o(s.v(hklv))
q = o(s.q(hklq))
p.points(v, c=c)
p.labels(v, map(format.tex_hex4, hex34v(hklv)), c=c)
p.circles(q, c=c)






plt.show()

