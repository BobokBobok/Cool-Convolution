import numpy as np
from numpy import loadtxt
import matplotlib.pyplot as plt
from scipy import signal

#Sawtooth function
t = np.linspace(0, 20, 1000, endpoint=True)
sawtooth = signal.sawtooth(0.5 * np.pi * t)

#Function for Cauchy distributed RV
def CauchyDist(x,Gamma):
    b = Gamma/2
    m = np.max(x)/2
    dx1 = np.array(abs(x[1]-x[0]))
    dxs = np.diff(x)
    dx = np.concatenate((np.array([dx1]),dxs))
    return 1/np.pi * (b/((x-m)**2+b**2)) * dx

def imfpConv(mu,a):
    IMFP = np.zeros((len(mu),len(mu))) #Allocate memory for square IMFP matrix
    #Generate matrix to convolve
    for i in range(0,len(mu)):
        IMFP[i,0:len(mu)] += mu #Rows are the gamma hole convolved data
            
    CONV = np.zeros((len(a),len(mu)))#Allocate memory for convolution matrix
        #Fill in convolution matrix
    for i in range(0,len(a)):
        y = CauchyDist(t,a[i]) #Each row of the convolution matrix is Cauchy distributed with width a(E)
            #print(a[i])
            #plt.plot(E,y)
        CONV[i,0:len(mu)] += y
            
            #Convolve the data row wise, this produces a matrix in which each row is a convoolution of the data due to a different width a(E). Return the diagonal of this matrix.
    CONVOLVED = signal.fftconvolve(IMFP,CONV,mode = 'same', axes = 1)
    return np.diagonal(CONVOLVED)
     
##Convolve sawtooth function with a Lorenzian
#Cauchy = CauchyDist(t,0.1)
#Convolution = signal.fftconvolve(Cauchy,sawtooth,mode='same')
# Plot the sawtooth wave
plt.plot(t,sawtooth, label="Original Sawtooth signal")
plt.plot(t,imfpConv(sawtooth,20/t), label="Convolved Sawtooth signal")
plt.xlabel("x")
plt.ylabel("y")
plt.title(r"Sawtooth convolved with Lorentzian of varying width $\Gamma = \frac{20}{x}$")
plt.legend()
#plt.axhline(y=0, color='black')
plt.show()
