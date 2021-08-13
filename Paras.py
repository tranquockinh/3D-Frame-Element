import numpy as np
#from StiffnessPlaneBeamBasic import StiffnessPlaneBeamBasic

# Defind the structure coordinates

ndcoords = 1.0 * np.matrix([[0.0, 0.0, 0.0],
                           [0.0, 3.0, 0.0],
                           [3.0, 3.0, 0.0],
                           [6.0, 3.0, 0.0],
                           [9.0, 0.0, 3.0]])
##print ('Coordinates: \n', ndcoords) 
# Defining nodal connection
elems = np.matrix([[0, 1],
                   [1, 2],
                   [2, 3],
                   [3, 4]])
##print ('Element connection: \n', elems) # checking
# Material properties
eparray =  1.0 * np.matrix([[200E6*0.01, 80E6*0.02, 200E6*0.001, 200E6*0.001],
                            [200E6*0.01, 80E6*0.02, 200E6*0.001, 200E6*0.001],
                            [200E6*0.01, 80E6*0.02, 200E6*0.001, 200E6*0.001],
                            [200E6*0.01, 80E6*0.02, 200E6*0.001, 200E6*0.001]])
##print ('Element properties: \n', eparray) # checking
# Initialization
ndim = 6
numnodes = np.size(ndcoords, 0) # number of elements in the first column
ndof = ndim * numnodes
nelems = np.size(elems, 0)
print ('number of element is \n ', nelems)
print ('Number of nodes is \n', numnodes) 

K = np.zeros((ndof, ndof),dtype=float)
ke = np.zeros((ndim * 2, ndim * 2), dtype=float)
##print ('K is \n {} \n and \n ke is \n {}'.format(K, ke)) # Checking

p = np.zeros((ndof, 1)) # to ensure the force vector equals ndof
peq = np.zeros((ndof, 1))
pe = np.zeros((ndim * 2, 1))
u = np.zeros((ndof,1))
##print (p,'\n',peq,'\n',pe,'\n','\n',u)

# Boundary conditions
bc = np.matrix([[0, 0, 0],
                [0, 1, 0],
                [0, 2, 0],
                [0, 3, 0],
                [0, 4, 0],
                [0, 5, 0],
                [4, 0, 0],
                [4, 1, 0],
                [4, 2, 0],
                [4, 3, 0],
                [4, 4, 0],
                [4, 5, 0]], dtype=int)
print (bc)                
cloads = np.matrix([[0,  0,                     0,                    0,   0,   0,   0],
                    [1,  0,                     0,                    0,   0,   0,   0],
                    [2, -240*np.cos(np.pi/4),  -240*np.cos(np.pi/4),  0,   0,   0,   0],
                    [3,  0,                    -60,                   0,   0,   0,  -180],
                    [4,  0,                     0,                    0,   0,   0,   0]])

print (cloads)

# Distributed load

qloads = np.matrix([[0,  40,  0.0, 0.0, 0.0, 0.0, 0.0],
                    [1,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [2,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [3,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

