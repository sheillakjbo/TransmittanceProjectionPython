import numpy as np
from numpy import linalg
from numpy.linalg import inv
import numpy.matlib
import math
import scipy as sci
import scipy.special
from scipy import linalg
from scipy.sparse.linalg import dsolve
import time

def Data(N,R,z0): 
        
        
#########################################################
###Random Uniform distribution of points in a cylinder###
#########################################################
    '''
    V = 2*np.pi*R*z0
    rho = N/(V) 
    r_min = 0 #rho ** (-1/3) / (0.8*np.pi)         #Minimal distance between two particles

    angle1 = 0
    angle2 = 2*np.pi



    x = np.empty([N,1])
    y = np.empty([N,1])
    z = np.empty([N,1])

    theta = (angle2 - angle1) * np.random.rand() + angle1
    r = R*np.sqrt(np.random.rand()) #R*np.random.rand() #
    x[0] = r*np.cos(theta)
    y[0] = r*np.sin(theta)
    z[0] = z0*np.random.rand() - z0/2


    nn=1
    for rr in range(0,100*N):
        theta = (angle2 - angle1) * np.random.rand() + angle1
        r = R*np.sqrt(np.random.rand())
        x1 = r*np.cos(theta)
        y1 = r*np.sin(theta)
        z1 = z0*np.random.rand() - z0/2
        
        r1 = np.sqrt((x1-x) ** 2 + (y1-y) ** 2 + (z1 -z)**2)     #Distance between the test poin and the other points
        test1 = r1 < r_min                          #Testing if the distance bewteen the points are large enough
        test = np.any(test1)


        if test == False:
            x[nn] = x1
            y[nn] = y1
            z[nn] = z1
            nn = nn + 1

        if nn == N:
            break
    '''
#########################################################
###Random Gaussian distribution of points in a cylinder###
#########################################################
    rhosq = np.random.uniform(0,R**2,N).reshape((N,1));
    rho1 = np.sqrt(rhosq);
    theta = np.random.uniform(0,2*np.pi,N).reshape((N,1));
    x = rho1*np.cos(theta);
    y = rho1*np.sin(theta);
    z = np.random.normal(-z0/2,z0/2,N).reshape((N,1));

    Repx = np.matlib.repmat(x,1,N)
    Repx1 = np.matlib.repmat(np.transpose(x),N,1)

    Data.DeltaX = Repx - Repx1

    Repy = np.matlib.repmat(y,1,N)
    Repy1 = np.matlib.repmat(np.transpose(y),N,1)

    Data.DeltaY = Repy - Repy1

    Repz = np.matlib.repmat(z,1,N)
    Repz1 = np.matlib.repmat(np.transpose(z),N,1)

    Data.DeltaZ = Repz - Repz1

    Data.D = np.sqrt(np.power(Data.DeltaX,2) + np.power(Data.DeltaY,2) + np.power(Data.DeltaZ,2))
    np.fill_diagonal(Data.D, 1)

    Data.x = x
    Data.y = y
    Data.z = z

    return

    


def EigVal(N,k0,EL1, EL1V, DeltaX, DeltaY, DeltaZ, D, Domega,vec):

    ###########################
    ### General Parameterns ###
    ###########################    
    Gamma_0 = 1
    Gamma_1 = Gamma_0/2              #Decay rate
    k = k0

    #Defining the number of modes:
    if vec == 1:
        NN = 3*N
    else:
        NN = N
    

    #########################################
    ###      Scalar Light Green Matrix    ###
    ######################################### 
    def MatrixS(Gamma_0,k,D,N,Domega1):
            
        H0 = (-Gamma_0/2) * np.exp(1j*k*D)/(1j*k*D)
        np.fill_diagonal(H0, 0)
        Diag = np.eye(N)
        HI = (1j * Domega1 - Gamma_0/2) * Diag
        return H0 + HI

    #from LinSysSttScattV2 import Data
    def MatrixV(Gamma_0,k0,D,N,Domega1):    
        #Green matrix: First portion of the sum
        Gd = 1j*(3*Gamma_0/4)*np.exp(1j*k0*D)/(k0*D)*(1 + 1j/(k0*D) - 1/( (k0*D)**2)) 
        np.fill_diagonal(Gd, 0)
        #Green matrix: Second portion of the sum
        Gt = 1j*(3*Gamma_0/4)*np.exp(1j*k0*D)/(k0*D)*(-1 - 3j/(k0*D) + 3/((k0*D)**2))/(D**2)
        np.fill_diagonal(Gt, 0)
        #Green Matrix: Positions product (ra*ra')
        XX = DeltaX**2
        YY = DeltaY**2
        ZZ = DeltaZ**2
        XY = DeltaX*DeltaY
        XZ = DeltaX*DeltaZ
        YZ = DeltaY*DeltaZ
        #Green Matrix: Diagonal
        Diag = np.eye(3*N)
        HI = (1j * Domega1 - Gamma_0/2) * Diag
        #Green Matrix: Assemblong Blocks
        Zr = np.zeros((N,N))
        GGd = np.block([[Gd, Zr, Zr],[Zr, Gd, Zr],[Zr, Zr, Gd]])
        GGt = np.block([[Gt, Gt, Gt],[Gt, Gt, Gt],[Gt, Gt, Gt]])
        XYZ = np.block([[XX, XY, XZ],[XY,YY, YZ],[XZ, YZ, ZZ]])

        return HI + (GGd + GGt*XYZ)            
 
    if vec == 0: 
        M0 = MatrixS(Gamma_0,k,D,N,Domega)
        E1 = -1j/2*EL1
        
    else:
        M0 = MatrixV(Gamma_0,k,D,N,Domega)
        E1 = -1j/2*EL1V
        

    
    W, V = linalg.eig(M0)
    EigVal.psi= V
    EigVal.lambd = W.reshape((NN,1))
   

    EigVal.alpha = -1*(sci.dot(inv(V),E1))/EigVal.lambd

    return


def LinSol(N,k0,EL1, EL1V, DeltaX, DeltaY, DeltaZ, D, Domega,vec):

    ###########################
    ### General Parameterns ###
    ###########################    
    Gamma_0 = 1
    k = k0

    #Defining the number of modes:
    if vec == 1:
        NN = 3*N
    else:
        NN = N
    

    #########################################
    ###      Scalar Light Green Matrix    ###
    ######################################### 
    def MatrixS(Gamma_0,k,D,N,Domega1):
            
        H0 = (-Gamma_0/2) * np.exp(1j*k*D)/(1j*k*D)
        np.fill_diagonal(H0, 0)
        Diag = np.eye(N)
        HI = (1j * Domega1 - Gamma_0/2) * Diag
        return H0 + HI

    #from LinSysSttScattV2 import Data
    def MatrixV(Gamma_0,k0,D,N,Domega1):    
        #Green matrix: First portion of the sum
        Gd = 1j*(3*Gamma_0/4)*np.exp(1j*k0*D)/(k0*D)*(1 + 1j/(k0*D) - 1/( (k0*D)**2)) 
        np.fill_diagonal(Gd, 0)
        #Green matrix: Second portion of the sum
        Gt = 1j*(3*Gamma_0/4)*np.exp(1j*k0*D)/(k0*D)*(-1 - 3j/(k0*D) + 3/((k0*D)**2))/(D**2)
        np.fill_diagonal(Gt, 0)
        #Green Matrix: Positions product (ra*ra')
        XX = DeltaX**2
        YY = DeltaY**2
        ZZ = DeltaZ**2
        XY = DeltaX*DeltaY
        XZ = DeltaX*DeltaZ
        YZ = DeltaY*DeltaZ
        #Green Matrix: Diagonal
        Diag = np.eye(3*N)
        HI = (1j * Domega1 - Gamma_0/2) * Diag
        #Green Matrix: Assemblong Blocks
        Zr = np.zeros((N,N))
        GGd = np.block([[Gd, Zr, Zr],[Zr, Gd, Zr],[Zr, Zr, Gd]])
        GGt = np.block([[Gt, Gt, Gt],[Gt, Gt, Gt],[Gt, Gt, Gt]])
        XYZ = np.block([[XX, XY, XZ],[XY,YY, YZ],[XZ, YZ, ZZ]])

        return HI + (GGd + GGt*XYZ)  
           
 
    if vec == 0: 
        M0 = MatrixS(Gamma_0,k,D,N,Domega)
        E = 1j/2*EL1
        
    else:
        M0 = MatrixV(Gamma_0,k,D,N,Domega)
        E = 1/2*EL1V
        

    
    sigma = np.linalg.solve(M0, E)    

    return sigma   