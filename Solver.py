from Paras import *
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
    u_reshaped = np.reshape(u,(numnodes,ndim)).T
    p_reshaped = np.reshape(p,(numnodes,ndim)).T
    return  u, p, u_reshaped, p_reshaped