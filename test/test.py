#!/usr/bin/python3

from empy import *

def test_grid():
    plt.figure(figsize=(12,6)) 

    p = Stereo(ax=plt.subplot(1,2,1))
    h = TiltRotation()
    h.grid(proj=p)
    h.range(proj=p)

    p = Stereo(ax=plt.subplot(1,2,2))
    h = DoubleTilt()
    h.grid(proj=p)
    h.range(proj=p)

def test_circles_points_and_labels():
    plt.figure(figsize=(12,6)) 

    p = Stereo(ax=plt.subplot(1,2,1))
    v = allv(2)
    p.points(v, s='auto')
    p.circles(v, c="r", lw='auto')

    p = Stereo(ax=plt.subplot(1,2,2))
    
    v = allv(4)
    p.points(v, s='auto', c=vrgb(v))
    
    v = allv(2)
    p.labels(v, color=vrgb(v))
    
    p.labels([(1,1,1),(0,1,1),(0,0,1),(0,0,-1)], ["A","B","C","X"], size=20)
    
    p.labels([(-1,-1,1),(0,-1,1)], format.tex, size=20)

    print(format.unicode((-10,-2,30)), format.unicode((-1,-2,-3)))

def test_hkl_allowed():
    hkls = [
        [1,0,0], [2,0,0],
        [1,1,0], [2,2,0],
        [1,1,1], [2,2,2],
        [1,1,2], [2,2,4],
    ]

    print("hkl", [format.simple(x) for x in hkls] )
    print("sc", structure.sc(1).hkl_allowed(hkls) )
    print("bcc", structure.bcc(1).hkl_allowed(hkls) )
    print("fcc", structure.fcc(1).hkl_allowed(hkls) )

def test_structures():
    plt.figure(figsize=(12,6)) 
    v = allv(3)
    
    s = structure.hcp(1,2)
    p = Stereo(ax=plt.subplot(1,2,1))

    qv = s.q(v)
    m = vlen(qv) <= 3*vlen(s.q([1,0,0]))

    p.points(qv[m], s='auto')
    p.circles(s.v(v), c='r')


    s = structure.fcc(1)
    o = s.orient([1,1,1],[1,-1,0])
    p = Stereo(ax=plt.subplot(1,2,2))
    
    p.points(o(s.q(v)), s='auto')
    p.circles(o(s.v(v)), c='r')

def test_laue():
    plt.figure(figsize=(12,6)) 
    
    s = structure.bcc(0.29)
    o = s.orient([1,2,3])
    
    p = Laue(ax=plt.subplot(1,2,1))
    v = allv(10)
    p.points(o(s.q(v[s.hkl_allowed(v)])),s='auto')
    v = allv(1)
    p.circles(o(s.v(v)),c='r')

    p = Laue(dir="forward", ax=plt.subplot(1,2,2))
    v = allv(10)
    p.points(o(s.q(v[s.hkl_allowed(v)])),s='auto')
    v = allv(1)
    p.circles(o(s.v(v)),c='r')


if __name__ == "__main__":
    
    test_grid()
    test_circles_points_and_labels()
    test_hkl_allowed()
    test_structures()
    test_laue()

    plt.show()
