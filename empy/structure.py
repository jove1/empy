
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

