#!/usr/bin/python3

from empy import cif
c = cif.read("Al.cif")
print( c['cell_length_a'] )
print( c['cell_length_b'] )
print( c['cell_length_c'] )
print( c['cell_angle_alpha'] )
print( c['cell_angle_beta'] )
print( c['cell_angle_gamma'])

from empy.structure import make_basis
print( make_basis(**c) )

for x,y,z,a in zip(
        c['atom_site_fract_x'],
        c['atom_site_fract_y'], 
        c['atom_site_fract_z'], 
        c['atom_site_type_symbol']):
        print( a,x,y,z )



from empy import format
print( format.tex([1,2,-3]) )
print( list(map(format.simple, [[1,-2,3],[2,2,2]])) )


from empy.structure import sym_expand
print( sym_expand([0,0,0], lambda x,y,z:[(-x,-y,z), (-x,y,-z), (z,x,y), (y,x,-z), (-x,-y,-z), (x,y+.5,z+.5), (x+.5,y,z+.5)]) )

from empy import Stereo, DoubleTilt, plt
p = Stereo()
h = DoubleTilt()
h.range(proj=p)
h.grid(proj=p)
plt.show()


