import math
import time

import numpy as np
#from numpy import linalg
import numpy.matlib
import scipy as sci
import scipy.special
from matplotlib import pyplot as plt
from scipy import linalg
from scipy.integrate import dblquad

vec = 0                         #vec~=1 => Scalat light If vec==1 => Vector light  \ro_lambda3 = 0.1;
ro_lambda3 = 0.1;       
#N = 20                        #Number of particles
steps = 13
lamb = 780*1e-9
k0 = 2*np.pi/lamb
k = k0
R = 4*lamb
x0 = 0                          #x size
y0 = 0
z0 = 1e-6 #np.linspace(2,10,steps).reshape((steps,1))                    #z size
N = np.round(ro_lambda3/lamb**3*(2*np.pi)**(1/2)*z0*np.pi*R**2); 
N = np.int_(N)
Gamma_0 = 1                     #Decay rate
Gamma_1 = Gamma_0/2             #Decay rate
Domega = np.linspace(-10,10,steps).reshape((steps,1))       # 1*np.ones((steps,1))*6#             #Single value of Domega (for tests)
W0 = 3*lamb
E0 = 1
Rlz = 10
rho = N/(2*np.pi*R*z0)
R0 = 50*R #  10*R+ (k*(W0**2))/2   #Distance of the sensors for far field approximation

##################### Adjusted optical thickness ######################
if vec==1:
     #b0 = 6*N./((R*k)^2);
     b0 = 6*N*sci.special.erf(1/np.sqrt(2))/((k*R)**2*np.sqrt(2*np.pi));#Normal z-axis distribution exact
     delt=2*Domega/Gamma_0;
     b=b0/(1+delt**2);
     #b=2.*b0./(16.*Domega.^2+1);   #Optical thickness of vector light

else:
    #b0 = 4*N./((R*k)^2); #Uniform distribution
    b0 = 4*N*sci.special.erf(1/np.sqrt(2))/((k*R)**2*np.sqrt(2*np.pi)); #Normal z-axis distribution exact
    delt=2*Domega/Gamma_0;
    b = b0/(1+delt**2);
    #b=b0./(4*Domega.^2+1);      #Optical thickness of scalar light




if vec == 1:
    NN = 3*N
else:
    NN = N


from FieldFunctions import *
from LinSys import *

ph = np.linspace(0.0, 2*np.pi, 400)
tht = np.linspace(0.0, np.pi/6, 400)


zr = np.pi*W0**2/lamb; # Rayleigh length
waist = W0*np.sqrt(1+(R0/zr)**2);
R1 = 5*waist;


start = time.time()
I_tot = np.zeros((steps,1))
for rr in range(0, Rlz):

    Data(N,R,z0)

    EL1 = LFF(Data.x,Data.y,Data.z,W0,E0,k)
        
    ELx = EL1/np.sqrt(2)           #Gaussian beam in the e+ direction
    ELy = EL1/np.sqrt(2)        #Gaussian beam in the e- direction
    ELz = np.zeros((N,1))               #Gaussian beam in the e- direction

    E1 = np.block([[ELx], [ELy], [ELz]])

    
    for dd in range(0,steps):
        
        Domega1 = Domega[dd]
        sigma = LinSol(N,k0, EL1, E1, Data.DeltaX, Data.DeltaY, Data.DeltaZ, Data.D, Domega1,vec)

        HalfRange = R1; 
        nn = 51;
        dl = 2*HalfRange/(nn-1);
        Xobsvec = np.linspace(-1,1,nn)*HalfRange;
        Yobsvec = Xobsvec;
        Xobs,Yobs = np.meshgrid(Xobsvec,Yobsvec);
        Zobs = R0*np.ones((nn,1));

        #Scalar
    
        EL= LFF(Xobs,Yobs,Zobs,W0,E0,k)

        Etot = EL+1j*Gamma_0*1/np.pi*TSFF(Data.x, Data.y, Data.z, Xobs, Yobs, Zobs, k, sigma)

        I_tot[dd]= I_tot[dd] + (np.absolute(np.sum(np.sum(np.conjugate(EL)*Etot*dl**2,axis=0)))/(np.pi/2*W0**2))**2/Rlz;

        zzz= np.conjugate(EL)*Etot*dl**2
        

        


end = time.time()
print(end - start)   



#'''
plot1 = plt.figure(1)
plt.plot(Domega, np.exp(-1*b),'k-') #, marker_size)
#plt.yscale('log')
plt.plot(Domega, I_tot, marker = 'o', markersize=5) #, marker_size)
#plt.yscale('log')
plt.title("Transmittance of Coherent Field")
plt.xlabel("$\Delta\omega$")
plt.ylabel('T')

plt.show()
#'''

