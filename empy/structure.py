
from .util import *

def make_basis(**kwargs):

        a = kwargs.get("cell_length_a", kwargs.get("a"))
        b = kwargs.get("cell_length_b", kwargs.get("b", a))
        c = kwargs.get("cell_length_c", kwargs.get("c", a))

        alpha = kwargs.get("alpha", kwargs.get("alpha",90))
        if a == b == c:
            beta = kwargs.get("cell_angle_beta", kwargs.get("beta", alpha))
            gamma = kwargs.get("cell_angle_gamma", kwargs.get("gamma", alpha))
        else:
            beta = kwargs.get("cell_angle_beta", kwargs.get("beta", 90))
            gamma = kwargs.get("cell_angle_gamma", kwargs.get("gamma", 90))
        
        Sa, Ca = sin(deg2rad(alpha)), cos(deg2rad(alpha))
        Sb, Cb = sin(deg2rad(beta)),  cos(deg2rad(beta))
        Sg, Cg = sin(deg2rad(gamma)), cos(deg2rad(gamma))
        
        CY = (Ca - Cg*Cb)/Sg
        CZ = sqrt(1 - Cb**2 - CY**2)
        
        basis = array([[  a,       0, 0],
                       [  b*Cg, b*Sg, 0],
                       [  c*Cb, c*CY, c*CZ]])
        
        if (alpha, beta, gamma) == (90, 90, 90):
            basis = np.diag([a,b,c])
        
        elif (alpha, beta, gamma) == (90, 90, 120) and a == b:
            basis = array([[-Cg*a, -Sg*a, 0],
                           [-Cg*b, +Sg*b, 0],
                           [    0,     0, c]])
        return basis

class Structure:

    def __init__(self, basis, sites, sym=None):
        self.basis = basis
        self.sites = [ (e,sym_expand(v,sym)) for e,v in sites ] if sym else sites

    def maxhkl(self, maxhkl_, sphere=False):
        vv = maxhkl(maxhkl_, sphere)
        return vv[self.hkl_allowed(vv)]

    def hkl_allowed(self, hkl, eps=1e-8):
        """return a mask of allowed reflections"""
        from numpy import zeros, sum, exp, pi, inner, abs, asarray
        hkl = asarray(hkl)
        r = zeros(hkl.shape[:-1], dtype=bool)
        for atom,site in self.sites:
            s = sum( exp( -2j*pi * inner(hkl, site) ), axis=-1)
            r |= (abs(s) > eps)
        return r

    def structf(self, hkl, atomff):
        from numpy import sum, exp, pi, inner
        qlen = vlen(self.q(hkl))
        fc = 0
        for atom,site in self.sites:
            fc += atomff(atom, qlen) * sum(exp( -2j*pi * inner(hkl, site) ), axis=-1)
        return fc

    def v(self, hkl):
        """miller index -> real space vector [hkl]"""
        from numpy import dot
        return dot(hkl, self.basis)

    def q(self, hkl):
        """miller index -> reciprocal space vector (hkl)"""
        from numpy import dot, linalg, pi
        return dot(hkl, linalg.inv(self.basis).T)

    def show3d(self, style={'A':1}):
        from mayavi import mlab
        scalar = []
        pos = []
        def rep(lst):
            for (x,y,z) in lst:
                yield x,y,z
                if x == 0:
                    yield 1,y,z
                    if y == 0:
                        yield 1,1,z
                if y == 0:
                    yield x,1,z
                    if z == 0:
                        yield x,1,1
                if z == 0:
                    yield x,y,1
                    if x == 0:
                        yield 1,y,1
                if x == y == z == 0:
                    yield 1,1,1

        for atom, site in self.sites:
            site = list(rep(site))
            scalar.extend([style[atom]]*len(site))
            pos.extend(self.v(site))
        scalar = np.asarray(scalar)
        pos = np.asarray(pos)


        mlab.points3d(pos[:,0],pos[:,1],pos[:,2],scalar,scale_factor=1, resolution=10)
        mlab.show()


    def show(self, style={'A':dict(c='r',s=100)}, n=2):
        border = [0,0],[1,0],[1,1],[0,1],[0,0]
        plt.figure()
        for i,(p,q) in enumerate([(0,1),(1,2),(2,0)]):
            ax = plt.subplot(2,2,1+i)
            ax.set_aspect('equal')
            ax.axis('off')
            for atom,site in self.sites:
                for shift in np.ndindex(n,n,n):
                    v = self.v(site) + self.v([shift])
                    plt.scatter(v[:,p],v[:,q],**style[atom])
                v = np.zeros((5,3))
                v[:,[p,q]] = border
                v = self.v(v)
                plt.plot(v[:,p],v[:,q])
        plt.tight_layout()

    def orient(self, zone=None, vec=None, dir=0):
        from numpy import sin, cos, pi, deg2rad
        if zone is None:
            zone, vec, dir = (0,0,1), (0,1,0), 0
        
        lst = [ ((0,0,1), self.v(zone)) ]
        if vec is not None:
            dir = deg2rad(dir)
            lst.append( ((cos(dir),sin(dir),0), self.q(vec)) )
       
        from .util import Orient
        return Orient.fit(lst)

_fcc0 = [(0,0,0), (.5,.5,0), (.5,0,.5), (0,.5,.5)]
_fcc1 = [(.25,.25,.25), (.75,.75,.25), (.75,.25,.75), (.25,.75,.75)] 
_fcc2 = [(.5,.5,.5), (0,0,.5), (0,.5,0), (.5,0,0)]
_fcc3 = [(.75,.75,.75), (.25,.25,.75), (.25,.75,.25), (.75,.25,.25)]

def fcc(a, A='A'):
    return Structure( np.diag([a,a,a]), sites=[(A, _fcc0)])

def bcc(a, A='A'):
    return Structure( np.diag([a,a,a]), sites=[(A, [(0,0,0),(.5,.5,.5)])])

def hcp(a, c=None, A='A'):
    if c is None:
        c = a*sqrt(8/3.)
    return Structure( 
            array([[.5*a, -sqrt(3/4.)*a, 0],
                   [.5*a, +sqrt(3/4.)*a, 0],
                   [   0,     0,         c]]), sites=[(A, [(0,0,0), (2/3.,1/3.,1/2.)]) ])

def sc(a, A="A"):
    return Structure( np.diag([a,a,a]), sites=[ (A, [(0,0,0)]) ])


"""

def diamond(a, A="A"):
    return Crystal(a=a, sites=[ (A, _fcc0+_fcc1) ])

def CsCl(a, A="A", B="B"):
    return Crystal(a=a, sites=[ (A, [(0,0,0)]),
                                (B, [(0.5,0.5,0.5)]) ])

def B32(a, A="A", B="B"):
    return Crystal(a=a, sites=[ (A, _fcc0+_fcc3), 
                                (B, _fcc1+_fcc2) ])

def D03(a, A="A", B="B"):
    return Crystal(a=a, sites=[ (A, _fcc0+_fcc1+_fcc2),
                                (B, _fcc3) ])

def NaCl(a, A="A", B="B"):
    return Crystal(a=a, sites=[ (A, _fcc0),
                                (B, _fcc2) ])

def ZnS(a, A="A", B="B"):
    return Crystal(a=a, sites=[ (A, _fcc0),
                                (B, _fcc1) ])

def fluorite(a, A="A", B="B"):
    return Crystal(a=a, sites=[ (A, _fcc0+_fcc2),
                                (B, _fcc1) ])

def half_heusler(a, A="A", B="B", C="C"):
    return Crystal(a=a, sites=[ (A, _fcc0),
                                (B, _fcc1),
                                (C, _fcc2) ])

def heusler(a, A="A", B="B", C="C"):
    return Crystal(a=a, sites=[ (A, _fcc0),
                                (B, _fcc1+_fcc3),
                                (C, _fcc2)])

# double sized cells

def sc_2(a, A="A"):
    return Crystal(a=2*a, sites=[ (A, _fcc0+_fcc2) ])

def bcc_2(a, A="A"):
    return Crystal(a=2*a, sites=[ (A, _fcc0+_fcc1+_fcc2+_fcc3) ])

def CsCl_2(a, A="A", B="B"):
    return Crystal(a=2*a, sites=[ (A, _fcc0+_fcc2), 
                                  (B, _fcc1+_fcc3) ])


# Strukturberricht names

Ah = sc
A1 = fcc
A2 = bcc
A3 = hcp
A4 = diamond

Ah_2 = sc_2
A2_2 = bcc_2
B2_2 = CsCl_2

B1 = NaCl
B2 = CsCl
B3 = ZnS

C1 = fluorite
C1b = half_heusler
L21 = heusler
"""
