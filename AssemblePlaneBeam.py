from Paras import *
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
    