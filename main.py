import numpy as np
from vpython import rate, gcurve, graph, canvas, color


# Field propreties and boundries :
Xmax = 6.5
Xmin = -6.5
Repartitions = 1000


# Generic constants :
hbar = 0.8 # Reduced Plank Constant
m = 1 # Particle Mass
p = 40 # Momentum
y0 = (p**2)/2*m # Height of the barrier 
x0 = 4 # Wave statring point
sig = 0.25 # Wave width
bw = 0.2 # barrier width


#Initial wave function ψ(x,0) :

def initial_packet(x,x0,sig,p):
    Psi0 = np.exp(-(x[1:-1]+x0)**2/sig**2
                )*np.exp(1j*p*(x[1:-1]+x0)
                        ) # Psi initialized and with two boundry elemnts less than x
    NormFact = np.sum(np.abs(Psi0)**2*dx) # Normalization factor 
    return(Psi0/np.sqrt(NormFact)) # Normalize Psi0

#Matrix for Hamiltonians :

def matrix_for_hamiltonians(Repartitions, dx, y, hbar, m):
    H = (hbar**2/(m*dx**2))*np.diag(np.ones(Repartitions-1)
                                    )+y[1:-1]*np.diag(np.ones(Repartitions-1)
                                                    )+ (-hbar**2/(2*m*dx**2)
                                                        )*np.diag(np.ones(Repartitions-2),1
                                                                    )+ (-hbar**2/(2*m*dx**2)
                                                                        )*np.diag(np.ones(Repartitions-2),-1)
    return H
 
'''
Creates a matrix with : 

Matrix Size: 
(999 x 999)

Main Diagonal(as A): 
(hbar^2 / (m * dx^2)) + y[1:-1]

Superdiagonal and Subdiagonal(as B): 
(-hbar^2 / (2 * m * dx^2)) 

Simplified 10x10 Example of the martrix shape :

 A   B   0   0   0   0   0   0   0   0
 B   A   B   0   0   0   0   0   0   0
 0   B   A   B   0   0   0   0   0   0
 0   0   B   A   B   0   0   0   0   0
 0   0   0   B   A   B   0   0   0   0
 0   0   0   0   B   A   B   0   0   0
 0   0   0   0   0   B   A   B   0   0
 0   0   0   0   0   0   B   A   B   0
 0   0   0   0   0   0   0   B   A   B
 0   0   0   0   0   0   0   0   B   A

'''
# Simulation setup
x = np.linspace(Xmin,Xmax,Repartitions+1)
dx = x[1]-x[0]
y = np.zeros_like(x)
y[(x > -bw/2) & (x < bw/2)] = y0

Psi0 = initial_packet(x,x0,sig,p)
H = matrix_for_hamiltonians(Repartitions, dx, y, hbar, m)

# Eigenvalues and eigenfunctions

E,psi = np.linalg.eigh(H) # Returns eigenvalues of that martix
psi = psi.T # Transposes the psi vector from columns to rows
NormFact2 = np.sum(np.abs(psi[0])**2*dx) # Normalization factor 
psi = psi/np.sqrt(NormFact2) # Normalize psi

'''
Using Vpython to plot dynamic animations
'''
scene = canvas()
g1 = graph(xtitle="x",ytitle="Re(ψ(x)) & Im(ψ(x))",width=1440, height=720)
f1 = gcurve(color=color.blue)
f2 = gcurve(color=color.green)
fy = gcurve(color=color.red)

# Plot barrier

for i in range(len(y)):
    fy.plot(x[i], 0.004 * y[i])

# Calculate coefficients
c = np.array([np.sum(np.conj(psi[i]) * Psi0 * dx) for i in range(len(psi))])

# Main time loop
t = 0
dt = 0.001
while t < 0.2:
    rate(24)
    Psi = np.sum(c[:, np.newaxis] * psi * np.exp(-1j * E[:, np.newaxis] * t / hbar), axis=0)
    f1.data = [[x[i], np.imag(Psi[i])] for i in range(len(Psi))]
    f2.data = [[x[i], np.real(Psi[i])] for i in range(len(Psi))]
    #f2.data = [[x[i], np.abs(Psi[i])] for i in range(len(Psi))] # use to visualize the absolute value of psi
    t += dt

