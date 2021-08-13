#Assemble the prescribed concentrated loads
from Paras import *
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
  