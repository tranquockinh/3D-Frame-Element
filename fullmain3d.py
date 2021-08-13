import numpy as np

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


# Computing stiffness matrix without considering of distributed loads
def StiffnessPlaneBeam(ex, ey, ez, ep, qloads):
    EA = ep[0, 0]
    GJ = ep[0, 1]
    EIy = ep[0, 2]
    EIz = ep[0, 3]
    b = np.matrix([[ex[1] - ex[0]],
                   [ey[1] - ey[0]],
                   [ez[1] - ez[0]]])
    
    L = float(np.sqrt(b.T * b))
    n = (1 / L) * b
    D = np.sqrt(n[0,0]**2 + n[1,0]**2)
    
    if D == 0:
        Lambda = np.matrix([[n[0,0],   n[1,0],   1.0],
                            [0.0   ,   1.0   ,   0.0],
                            [-1.0  ,   0.0   ,   0.0]])
    else:
        Lambda = np.matrix([[n[0,0],               n[1,0],             n[2,0]],
                            [-n[1,0]/D,            n[0,0]/D,           0.0],
                            [-n[0,0]*n[2,0]/D,    -n[0,0]*n[1,0]/D,    D]])
    # Compute the transfomation matrix T for coordinate transformation 
    #in 2D space 
    
    T = np.zeros((2*ndim,2*ndim))
    T[0:3,0:3] = Lambda
    T[3:6,3:6] = Lambda
    T[6:9,6:9] = Lambda
    T[9:12,9:12] = Lambda
    
    lload = np.matrix([[qloads[0,0]],
                       [qloads[0,1]],
                       [qloads[0,2]],
                       [qloads[0,3]],
                       [qloads[0,4]],
                       [qloads[0,5]]])
                       
    qxle = float(T[0, :6] * lload)
    qyle = float(T[1, :6] * lload)
    qzle = float(T[2, :6] * lload)
    mxle = float(T[3, :6] * lload)
    myle = float(T[4, :6] * lload)
    mzle = float(T[5, :6] * lload)
    ## Compute local distributed load matrix
    fle = np.matrix([[qxle * L / 2],
                     [qyle * L / 2],
                     [qzle * L / 2],
                     [-mxle * L / 2],
                     [qzle * L**2 / 12 - myle * L / 2],
                     [qyle * L**2 / 12 - mzle * L / 2],
                     [qxle * L / 2],
                     [qyle * L / 2],
                     [qzle * L / 2],
                     [mxle * L / 2],
                     [-qzle * L**2 /12 + myle * L / 2],
                     [-qyle * L**2 / 12 + mzle * L / 2]])
                     
                     ## Compute the global load vector
    pe = np.transpose(T) * fle
                   
    # Local element stiffness matrix initialization
    
    kle = np.zeros((ndof, ndof),dtype=float)
    kaa_EA = EA/L
    kaa_GJ = GJ/L
    kvvy = 12*EIy/L**3
    kvmy = 6*EIy/L**2
    kvvz = 12*EIz/L**3
    kvmz = 6*EIz/L**2
    kmmy = 4*EIy/L
    kmvy = 2*EIy/L
    kmmz = 4*EIz/L
    kmvz = 2*EIz/L 
   
    # Local element stiffness matrix
    kle = np.matrix(
    [[kaa_EA,  0,     0,     0,       0,    0,    -kaa_EA,  0,      0,      0,      0,       0],
     [0,       kvvz,  0,     0,       0,    kvmz,  0,      -kvvz,   0,      0,      0,       kvmz],
     [0,       0,     kvvy,  0,     -kvmy,  0,      0,       0,     -kvvy,   0,     -kvmy,    0],
     [0,       0,     0,     kaa_GJ, 0,     0,     0,       0,      0,     -kaa_GJ, 0,       0],
     [0,       0,    -kvmy,   0,      kmmy,  0,     0,       0,      kvmy,   0,      kmvy,    0],
     [0,       kvmz,  0,     0,      0,     kmmz,  0,      -kvmz,   0,      0,      0,       kmvz],
     [-kaa_EA, 0,     0,     0,      0,     0,     kaa_EA,  0,      0,      0,      0,       0],
     [0,      -kvvz,  0,     0,      0,    -kvmz,  0,       kvvz,   0,      0,      0,      -kvmz],
     [0,       0,    -kvvy,  0,      kvmy,  0,     0,       0,      kvvy,   0,      kvmy ,   0],
     [0,       0,     0,    -kaa_GJ, 0,     0,     0,       0,      0,      kaa_GJ, 0,       0],
     [0,       0,    -kvmy,  0,      kmvy,  0,     0,       0,      kvmy,   0,      kmmy,    0],
     [0,       kvmz,  0,     0,      0,     kmvz,  0,      -kvmz,   0,      0,      0,       kmmz]])
    
    Ke = np.transpose(T) * kle * T
    
    return Ke, T, kle, pe, L, b   

# Computing stiffness matrix with considering of distributed loads    
def StiffnessPlaneBeam(ex, ey, ez, ep, qloads):
    EA = ep[0, 0]
    GJ = ep[0, 1]
    EIy = ep[0, 2]
    EIz = ep[0, 3]
    b = np.matrix([[ex[1] - ex[0]],
                   [ey[1] - ey[0]],
                   [ez[1] - ez[0]]])
    
    L = float(np.sqrt(b.T * b))
    n = (1 / L) * b
    D = np.sqrt(n[0,0]**2 + n[1,0]**2)
    
    if D == 0:
        Lambda = np.matrix([[n[0,0],   n[1,0],   1.0],
                            [0.0   ,   1.0   ,   0.0],
                            [-1.0  ,   0.0   ,   0.0]])
    else:
        Lambda = np.matrix([[n[0,0],               n[1,0],             n[2,0]],
                            [-n[1,0]/D,            n[0,0]/D,           0.0],
                            [-n[0,0]*n[2,0]/D,    -n[0,0]*n[1,0]/D,    D]])
    # Compute the transfomation matrix T for coordinate transformation 
    #in 2D space 
    
    T = np.zeros((2*ndim,2*ndim))
    T[0:3,0:3] = Lambda
    T[3:6,3:6] = Lambda
    T[6:9,6:9] = Lambda
    T[9:12,9:12] = Lambda
    
    lload = np.matrix([[qloads[0,0]],
                       [qloads[0,1]],
                       [qloads[0,2]],
                       [qloads[0,3]],
                       [qloads[0,4]],
                       [qloads[0,5]]])
                       
    qxle = float(T[0, :6] * lload)
    qyle = float(T[1, :6] * lload)
    qzle = float(T[2, :6] * lload)
    mxle = float(T[3, :6] * lload)
    myle = float(T[4, :6] * lload)
    mzle = float(T[5, :6] * lload)
    ## Compute local distributed load matrix
    fle = np.matrix([[qxle * L / 2],
                     [qyle * L / 2],
                     [qzle * L / 2],
                     [-mxle * L / 2],
                     [qzle * L**2 / 12 - myle * L / 2],
                     [qyle * L**2 / 12 - mzle * L / 2],
                     [qxle * L / 2],
                     [qyle * L / 2],
                     [qzle * L / 2],
                     [mxle * L / 2],
                     [-qzle * L**2 /12 + myle * L / 2],
                     [-qyle * L**2 / 12 + mzle * L / 2]])
                     
                     ## Compute the global load vector
    pe = np.transpose(T) * fle
                   
    # Local element stiffness matrix initialization
    kle = np.zeros((ndof, ndof),dtype=float)
    kaa_EA = EA/L
    kaa_GJ = GJ/L
    kvvy = 12*EIy/L**3
    kvmy = 6*EIy/L**2
    kvvz = 12*EIz/L**3
    kvmz = 6*EIz/L**2
    kmmy = 4*EIy/L
    kmvy = 2*EIy/L
    kmmz = 4*EIz/L
    kmvz = 2*EIz/L 
   
    # Local element stiffness matrix
    kle = np.matrix(
    [[kaa_EA,  0,     0,     0,       0,    0,    -kaa_EA,  0,      0,      0,      0,       0],
     [0,       kvvz,  0,     0,       0,    kvmz,  0,      -kvvz,   0,      0,      0,       kvmz],
     [0,       0,     kvvy,  0,     -kvmy,  0,      0,       0,     -kvvy,   0,     -kvmy,    0],
     [0,       0,     0,     kaa_GJ, 0,     0,     0,       0,      0,     -kaa_GJ, 0,       0],
     [0,       0,    -kvmy,   0,      kmmy,  0,     0,       0,      kvmy,   0,      kmvy,    0],
     [0,       kvmz,  0,     0,      0,     kmmz,  0,      -kvmz,   0,      0,      0,       kmvz],
     [-kaa_EA, 0,     0,     0,      0,     0,     kaa_EA,  0,      0,      0,      0,       0],
     [0,      -kvvz,  0,     0,      0,    -kvmz,  0,       kvvz,   0,      0,      0,      -kvmz],
     [0,       0,    -kvvy,  0,      kvmy,  0,     0,       0,      kvvy,   0,      kvmy ,   0],
     [0,       0,     0,    -kaa_GJ, 0,     0,     0,       0,      0,      kaa_GJ, 0,       0],
     [0,       0,    -kvmy,  0,      kmvy,  0,     0,       0,      kvmy,   0,      kmmy,    0],
     [0,       kvmz,  0,     0,      0,     kmvz,  0,      -kvmz,   0,      0,      0,       kmmz]])
    
    Ke = np.transpose(T) * kle * T    
    return Ke, T, kle, pe, L, b   

def AssemblePlaneBeam(K, Ke, LeftNode, RightNode):

    LeftDOFu = LeftNode * ndim
    LeftDOFv = LeftDOFu + 1
    LeftDOFw = LeftDOFu + 2
    LeftDOFth1 = LeftDOFu + 3
    LeftDOFth2 = LeftDOFu + 4
    LeftDOFth3 = LeftDOFu + 5
    RightDOFu = RightNode * ndim
    RightDOFv = RightDOFu + 1
    RightDOFw = RightDOFu + 2
    RightDOFth1 = RightDOFu + 3
    RightDOFth2 = RightDOFu + 4
    RightDOFth3 = RightDOFu + 5   
    StiffnessGlobal = np.array([LeftDOFu,LeftDOFv,LeftDOFw,LeftDOFth1,LeftDOFth2,LeftDOFth3, \
                                RightDOFu,RightDOFv,RightDOFw,RightDOFth1,RightDOFth2,RightDOFth3])
    print('Global element stiffness index array:', StiffnessGlobal)
    # Append new values into initial global stiffness matrix
    for i in StiffnessGlobal:
        for j in StiffnessGlobal:
            m = np.array(np.where(StiffnessGlobal==i)).flatten()[0]
            n = np.array(np.where(StiffnessGlobal==j)).flatten()[0]
            ##K[i,j] = K[i,j] + Ke[i-LeftDOFu,j-LeftDOFu]
            K[i,j] = K[i,j] + Ke[m,n]
    
    return K
    
#Assemble the prescribed concentrated loads
def AssemblePlaneBeamForces(p, cloads):

    # p is a vector size (ndof,1)
    # cloads is a matrix defined with loading components on loaded 
    #node numbers   
    numLoadedNodes = np.size(cloads, 0) ## number of nodes loaded
    
    for i in range(numLoadedNodes):
        ## Pull load components of each current node
        nodenum = cloads[i, 0]
        px = cloads[i, 1]
        py = cloads[i, 2]
        pz = cloads[i, 3]
        mx = cloads[i, 4]
        my = cloads[i, 5]
        mz = cloads[i, 6]
        pNode = np.array([px, py, pz, mx, my, mz])
        ## pass to global concentrated load vector
        ## Locate the coordinates of global loading components
        pxDof = nodenum * ndim
        pyDof = nodenum * ndim + 1
        pzDof = nodenum * ndim + 2
        mxDof = nodenum * ndim + 3
        myDof = nodenum * ndim + 4
        mzDof = nodenum * ndim + 5       
        pDof = np.array([pxDof, pyDof, pzDof, mxDof, myDof, mzDof], dtype=int)
        ##print('pDof is:\n', pDof)
        ## Append the load components of each node to the right coordinates
        ## p is definded above at the beginning
        for j in pDof:
            p[j] = p[j] + pNode[j-int(pxDof)]   
    return p    
p = AssemblePlaneBeamForces(p, cloads)
print('Global force vector is p = \n {}'.format(p))    
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
def Solver(K, u, p, peq, bc):
    numPrescribedDof = np.size(bc, 0)
    numFreeDof = ndof - numPrescribedDof
    idxPrescribedDof = np.zeros((numPrescribedDof, 1), dtype=int)
    idxFreeDof = np.zeros((numFreeDof, 1), dtype=int)
    
    ## Index of prescribed DOF
    for i in range(numPrescribedDof):
        nodePrescibed = bc[i, 0]
        xyzORtheta123 = bc[i, 1]
        NumericalValue = bc[i, 2]
        idxPrescribed = nodePrescibed * ndim + xyzORtheta123
        u[idxPrescribed] = NumericalValue
        idxPrescribedDof[i] = idxPrescribed
    
    ## Index of free DOF
    FreeDof = np.arange(0,ndof)
    FreeDof = np.delete(FreeDof, idxPrescribedDof)
    
    ## Compute global displacement and global nodal forces
    ## Initialization
    u_compute = np.zeros((np.size(FreeDof),1), dtype=float)
    p_prescribed = np.zeros((np.size(idxPrescribedDof),1), dtype=float)
    K_compute = np.zeros((np.size(FreeDof),np.size(FreeDof)))
    K_prescribed = np.zeros((np.size(idxPrescribedDof),np.size(idxPrescribedDof)))
    K_presFree = np.zeros((np.size(idxPrescribedDof),np.size(FreeDof)))
    K_Freepres = np.zeros((np.size(FreeDof),np.size(idxPrescribedDof)))
    p_compute = np.zeros((np.size(FreeDof),1), dtype=float)
    peq_compute = np.zeros((np.size(FreeDof),1), dtype=float)
    peq_prescribed = np.zeros((np.size(idxPrescribedDof),1), dtype=float)
    u_Prescribed = np.zeros((np.size(idxPrescribedDof),1), dtype=float)           
    
    for j in FreeDof:
        for k in FreeDof:   
            m = np.array(np.where(FreeDof == j)).flatten()[0]
            n = np.array(np.where(FreeDof == k)).flatten()[0]
            #u_Prescribed[m] = u[j]
            p_compute[m] = p[j]
            peq_compute[m] = peq[j]
            K_compute[m, n] = K[j, k]
            
    for j in idxPrescribedDof:
        for k in idxPrescribedDof:
            m = np.array(np.where(idxPrescribedDof == j)).flatten()[0]
            n = np.array(np.where(idxPrescribedDof == k)).flatten()[0]
            u_Prescribed[m] = u[j]
            #p_prescribed[m] = p[j]
            peq_prescribed[m] = peq[j]
            K_prescribed[m,n] = K[j,k]
            
    for j in idxPrescribedDof:
        for k in FreeDof:
            m = np.array(np.where(idxPrescribedDof == j)).flatten()[0]
            n = np.array(np.where(FreeDof == k)).flatten()[0]
            K_presFree[m,n] = K[j,k]
            K_Freepres[n,m] = K[k,j]
    ## Check symmetry of the K compute                
    for i in range(np.size(FreeDof)):
            for j in range(np.size(FreeDof)):
                roundoff_error = 1.0E-12
                if i != j:
                    if (K_compute[i, j] - K_compute[j, i]) > roundoff_error:                    
                        print ('Check symmetric matrix')
                    else: continue
    
    u_compute = np.matmul((np.linalg.inv(K_compute)), \
    (p_compute + peq_compute -  np.matmul(K_Freepres, u_Prescribed)))    
    
    p_prescribed = np.matmul(K_presFree,u_compute) + \
    np.matmul(K_prescribed,u_Prescribed) - peq_prescribed
    
    for i, j in enumerate(FreeDof):
        u[j] = u_compute[i]
    for i,j in enumerate(idxPrescribedDof):
        p[j] = p_prescribed[i]
    
    return  u, p

u, p = Solver(K, u, p, peq, bc)

print('Dislacement is U: \n', u)
print()
print('Global force vector p: \n', p)