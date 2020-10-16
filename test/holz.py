#!/usr/bin/python3

from empy import *


s = structure.bcc(.29)
v = maxhkl(20, True)
v = v[s.hkl_allowed(v)]




o = s.orient([1,1,0],[0,0,1])
lz = dot(v, [1,1,0])

vv = o(s.q(v))
ee = ex_error(vv)

xi = 70
m = 1/hypot(1,ee*xi) > 0.01
p = Flat()
#p = ThreeD()

p.points(vv[m], 
        c=lz[m],
        #c='k', 
        s=40/hypot(1,ee*xi)[m]
    )
plt.grid()



o = s.orient([6,4,1],[-1,1,2])
#o = s.orient([1,1,0])
lz = dot(v,[1,1,0]) 

vv = o(s.q(v))
ee = ex_error(vv)
xi = 7
m = 1/hypot(1,ee*xi) > 0.01
m &= abs(lz) <= 1

p = Flat()
p.points(vv[m],
        c=lz[m],
        s=40/hypot(1,ee*xi)[m]
    ) 
p.equal_aspect(35)


# Hirsh example p. 115
s = structure.fcc(.29)

v = maxhkl(5)
v = v[s.hkl_allowed(v)]

z = [2,1,1]
o = s.orient(z,[0,-1,1])
lz = dot(v, z)
print( np.unique(lz) )
m = (lz>=0)&(lz<=2)

vv = o(s.q(v))

p = Flat()
plt.title("Hirsh example p.115")
p.points(vv[m], c=lz[m])
p.labels(vv[m], map(format.simple, v[m]))
p.points(o(s.q([[-2,-1,5]]))/3., c="c")
p.equal_aspect(25)


plt.show()
