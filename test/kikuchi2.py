#!/usr/bin/python3

from empy import *
phase = structure.hcp(0.323,0.514)
hkl = maxhkl(10)
hklq = hkl[(vlen(phase.q(hkl))<10) & phase.hkl_allowed(hkl)]
hklv = hkl[vlen(phase.v(hkl))<1.0]

o = phase.orient([1,0,1],[0,1,0],90)
q = o(phase.q(hklq))
v = o(phase.v(hklv))

p = Stereo(ax=a4figure("landscape").add_axes([0,0,1,1]))
p.ax.cla()
p.ax.grid(False)
p.ax.axis("off")
p.kikuchi(q)


h,k,l = hklv.T
hkl4 = np.column_stack([2*h-k, 2*k-h, -(h+k), 3*l])
hkl4 //= np.gcd.reduce(hkl4, axis=-1, keepdims=True)

p.labels(v, list(map(lambda x: "["+format.tex_hex4(x)+"]", hkl4)))
#p.points(v)


p.ax.set_aspect("equal")
p.ax.set_xlim(-0.5,0.66)
p.ax.set_ylim(-0.41,0.41)

plt.savefig("hex.pdf")
plt.show()
