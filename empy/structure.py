

class Crystal:
    def __init__(self, **kwargs):
        from numpy import sin, cos, pi, array, diag, sqrt

        self.sites = kwargs.get("sites", [])

        a = kwargs.get("a")
        b = kwargs.get("b", a)
        c = kwargs.get("c", a)

        alpha = kwargs.get("alpha", 90)
        if a == b == c:
            beta = kwargs.get("beta", alpha)
            gamma = kwargs.get("gamma", alpha)
        else:
            beta = kwargs.get("beta", 90)
            gamma = kwargs.get("gamma", 90)
        
        Sa, Ca = sin(pi*alpha/180.), cos(pi*alpha/180.)
        Sb, Cb = sin(pi*beta/180.), cos(pi*beta/180.)
        Sg, Cg = sin(pi*gamma/180.), cos(pi*gamma/180.)
        
        CY = (Ca - Cg*Cb)/Sg
        CZ = sqrt(1 - Cb**2 - CY**2)
        
        self.system = "triclinic"
        self.basis = array([[  a,       0, 0],
                            [  b*Cg, b*Sg, 0],
                            [  c*Cb, c*CY, c*CZ]])
        
        if alpha == beta == gamma == 90:
            if a == b == c:
                self.system = "cubic"
            elif a == b or b == c or c == a:
                self.system = "tetragonal"
            else:
                self.system = "orthorhombic"
            self.basis = diag([a,b,c])
        
        elif alpha == beta == 90 and a==b and  gamma == 120:
            self.system = "hexagonal"
            self.basis = array([[-Cg*a, -Sg*a, 0],
                                [-Cg*b, +Sg*b, 0],
                                [    0,     0, c]])

        elif beta == gamma == 90 and b==c and  alpha == 120:
            self.system = "hexagonal"
            self.basis = array([[    0,     0, a],
                                [-Ca*b, -Sa*b, 0],
                                [-Ca*c, +Sa*c, 0]])
        
        elif gamma == alpha == 90 and c==a and  beta == 120:
            self.system = "hexagonal"
            self.basis = array([[-Cb*a, +Sb*a, 0],
                                [    0,     0, b],
                                [-Cb*c, -Sb*c, 0]])
        
        elif beta == gamma == 90  or alpha == gamma == 90 or alpha == beta == 90:
            self.system = "monoclinic"
        
        elif a == b == c and alpha == beta == gamma:
            # is there a special symetric basis in this case?
            self.system = "trigonal"

    def hkl_allowed(self, hkl):
        """return a mask of allowed reflections"""
        from numpy import zeros, sum, exp, pi, inner, abs, asarray
        hkl = asarray(hkl)
        r = zeros(hkl.shape[:-1], dtype=bool)
        for atom,site in self.sites:
            s = sum( exp( -2j*pi * inner(hkl, site) ), axis=-1)
            r |= (abs(s) > 1e-10)
        return r

    def v(self, hkl, rot=True):
        """miller index -> real space vector"""
        from numpy import dot
        return dot(hkl, self.basis)

    def q(self, hkl, rot=True):
        """miller index -> reciprocal space vector"""
        from numpy import dot, linalg, pi
        return dot(hkl, 2 * pi * linalg.inv(self.basis).T)
    
    def orient(self, zone=None, vec=None, dir=0):
        from numpy import sin, cos, pi
        if zone is None:
            zone, vec, dir = (0,0,1), (0,1,0), 0
        
        lst = [ ((0,0,1), self.v(zone)) ]
        if vec is not None:
            lst.append( ((cos(dir/180.*pi),sin(dir/180.*pi),0), self.q(vec)) )
       
        from .util import Orient
        return Orient.fit(lst)

_fcc0 = [(0,0,0), (.5,.5,0), (.5,0,.5), (0,.5,.5)]
_fcc1 = [(.25,.25,.25), (.75,.75,.25), (.75,.25,.75), (.25,.75,.75)] 
_fcc2 = [(.5,.5,.5), (0,0,.5), (0,.5,0), (.5,0,0)]
_fcc3 = [(.75,.75,.75), (.25,.25,.75), (.25,.75,.25), (.75,.25,.25)]

def hcp(a, c, A="A"):
    return Crystal(a=a, c=c, gamma=120, sites=[ (A, [(0,0,0), (2/3.,1/3.,1/2.)]) ])

def sc(a, A="A"):
    return Crystal(a=a, sites=[ (A, [(0,0,0)]) ])

def fcc(a, A="A"):
    return Crystal(a=a, sites=[ (A, _fcc0) ])

def bcc(a, A="A"):
    return Crystal(a=a, sites=[ (A, [(0,0,0), (.5,.5,.5)]) ])

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

