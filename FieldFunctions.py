import numpy as np
import scipy as sp
import math

# Laser Field Function in a vacuum
def LFF(xl,yl,zl,W0,E0,k):

    alpha = 1 + 2*1j*zl/(k*(W0**2))    
    EL = E0*np.exp(1j*k*zl)/(alpha)*np.exp(-1*(xl**2 + yl**2)/(alpha*(W0**2)))
   
    return EL

    


#TSFF - Total Scattered Field Function Scalar
def TSFF(x, y, z, x1, y1, z1, k, sigma2):

    num_rows, num_cols = x.shape
    Drrj = 0

    
    #'''
    EScatt = 0 
    for ii in range(0, num_rows):
        #print(x1.shape)
        Drrj = np.sqrt( (x1-x[ii])**2 + (y1-y[ii])**2 + (z1-z[ii])**2)
        EScatt = EScatt - np.exp(1j*k*Drrj)/(1j*k*Drrj)*sigma2[ii]

    #'''

    '''
    Drrj = np.sqrt( (x1-x)**2 + (y1-y)**2 + (z1-z)**2)
    EScatt = -1*np.matmul(np.transpose(np.exp(1j*k*Drrj)/(1j*k*Drrj) ),sigma)[0][0]
    '''
    return EScatt


    

#TSFF - Total Scattered Field FUnction Vector
def TSFFV(x1,y1,z1,x_r1,y_r1,z_r1,k0,sigma1, Gamma_0,R0):

    N = np.size(x_r1,0)


    
    TSFFV.Escattx = 0; TSFFV.Escatty = 0; TSFFV.Escattz = 0; #God = 0; Got = 0; xoj = 0; Roj = 0; Got = 0;
    for ii in range(0,N):
        
        xoj = x1-x_r1[ii]; yoj = y1-y_r1[ii]; zoj = z1-z_r1[ii]
        Roj = np.sqrt(xoj**2 + yoj**2 + zoj**2)

        God = 3/4*np.exp(1j*k0*Roj)/(k0*Roj)*(1 + 1j/(k0*Roj) - 1/((k0*Roj)**2))
        
        #Green matrix: Second portion of the sum
        Got = 3/4*np.exp(1j*k0*Roj)/(k0*Roj)*(-1 - 3j/(k0*Roj) + 3/((k0*Roj)**2))/(Roj**2)
          
        
        TSFFV.Escattx = TSFFV.Escattx - ((God + Got*xoj**2)*sigma1[ii]\
            + Got*xoj*yoj*sigma1[N+ii]\
            + Got*xoj*zoj*sigma1[2*N+ii])
        
        TSFFV.Escatty = TSFFV.Escatty - (Got*xoj*yoj*sigma1[ii]\
            + (God + Got*yoj**2)*sigma1[N+ii]\
            + Got*zoj*yoj*sigma1[2*N+ii])
       
        TSFFV.Escattz = TSFFV.Escattz - (Got*xoj*zoj*sigma1[ii]\
            + Got*yoj*zoj*sigma1[N+ii]\
            + (God + Got*zoj**2)*sigma1[2*N+ii])


    return [TSFFV.Escattx, TSFFV.Escatty, TSFFV.Escattz]


