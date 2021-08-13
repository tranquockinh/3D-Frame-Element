from Paras import *
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
