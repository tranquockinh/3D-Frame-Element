import numpy as np
from Paras import *
from StiffnessPlaneBeamBasic import StiffnessPlaneBeamBasic
from StiffnessPlaneBeam import StiffnessPlaneBeam
from AssemblePlaneBeam import AssemblePlaneBeam
from AssemblePlaneBeamForces import AssemblePlaneBeamForces
from Solver import Solver

p = AssemblePlaneBeamForces(p, cloads)  
#print('Global force vector is p = \n {}'.format(p))    

# Assemble stiffness matrix
for ThisElement in range(nelems):
    LeftNode = elems[ThisElement, 0]
    RightNode = elems[ThisElement, 1]
    ex = [ndcoords[LeftNode, 0], ndcoords[RightNode, 0]]
    ey = [ndcoords[LeftNode, 1], ndcoords[RightNode, 1]]
    ez = [ndcoords[LeftNode, 2], ndcoords[RightNode, 2]]
    ep = eparray[ThisElement, :]
    
    ## Take care of distributed loads
    found = 0 ## suppose there are no distributed loads applied
    for i in range(np.size(qloads,0)):
        if ThisElement == qloads[i,0]:
            found = 1
            break
    if found:
        Ke, T, kle, pe, L, b = StiffnessPlaneBeam(ex, ey, ez, ep, qloads[i, 1:])      
        pe = np.array(pe.T).flatten() ## Fattenning pe to an array
        peLeft = np.append([LeftNode], pe[:6])
        peRight = np.append([RightNode], pe[6:])
        eqvcloads = np.matrix([peLeft, peRight])
        print('Equivalent loads:\n', eqvcloads)
        print(eqvcloads.dtype)
        ## Equivalent loads
        peq = AssemblePlaneBeamForces(peq, eqvcloads)
    else:        
        ## Call the function of stiffness matrix
        Ke,T, kle, L, b = StiffnessPlaneBeamBasic(ex, ey, ez, ep) 
        
        ## Check symmetry of the stiffness matrix
        for i in range(2 * ndim):
            for j in range(2 * ndim):
                roundoff_error = 1.0E-12
                if i != j:
                    if (Ke[i, j] - Ke[j, i]) > roundoff_error:                    
                        print ('Check symmetric matrix')
                    else: continue
    ## Assemble plane beam
    K = AssemblePlaneBeam(K, Ke, LeftNode, RightNode)
    ## Check symmetry of the stiffness matrix
    for i in range(ndof):
            for j in range(ndof):
                roundoff_error = 1.0E-12
                if i != j:
                    if (K[i, j] - K[j, i]) > roundoff_error:                    
                        print ('Check symmetric matrix')
                    else: continue
    ## Check local striffness matrix
    print(' \
    Global element stiffness matrix of element Ke{} =\n {} \n \
    transformation matrix T{} =\n {}'.format(ThisElement,Ke,ThisElement,T)
    ) ## Check global stiffness matrix
    print('Global stiffness matrix of element {} is K =\n {}'. \
    format(ThisElement, K))
    print('Shape of K is: \n', np.shape(K))
    print('length of element L{} is\n {}\n and b{} is\n {} '. \
    format(ThisElement,L,ThisElement,b))
    print(ex,ey,ep)
    print()
    ##print('Ke{}=\n{}'.format(ThisElement, np.transpose(T) * kle * T))
# Solving the displacement and global force vector
u, p, u_reshaped, p_reshaped = Solver(K, u, p, peq, bc)

print('Dislacement is U: \n', u)
print()
print('Global force vector p: \n', p)