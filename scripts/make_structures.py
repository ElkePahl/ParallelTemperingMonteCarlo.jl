
from ase.build import bulk
from ase.visualize import view

# Create an HCP structure for Argon
hcp_Ar = bulk('Ar', 'hcp', a=5.256, c=8.492, orthorhombic=True)

# Extend the unit cell to get 27 atoms (3x3x3 supercell)
hcp_Ar_3x3x3 = hcp_Ar.repeat((3, 3, 3))

# Visualize the structure
view(hcp_Ar_3x3x3)
