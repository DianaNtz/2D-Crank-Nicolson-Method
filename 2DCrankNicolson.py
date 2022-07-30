"""
@author: Diana Nitzschke
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import cm
import os
import imageio
hbar = 1.0545718e-34
mass = 9.11e-31         # mass electron
wx = 110*2*np.pi*1000   # freuquency in Hz
wy = 110*2*np.pi*1000   # freuquency in Hz
x0 = np.sqrt(hbar / (wx * mass)) # in m
y0 = np.sqrt(hbar / (wy * mass)) # in m
dt = 1e-8  # dt in s
tfinal = 2*np.pi/wx # final time in s
time_steps = int(np.round(tfinal/dt))
t = 0.0  # initial time
N = 80
xmax = 6.9 * x0
xmin = -6.9 * x0
ymax = 4.6 * y0
ymin = -4.6 * y0
dx = (xmax - xmin) / ((N - 1))
dy = (ymax - ymin) / ((N - 1))
xn=np.linspace(-N/2,N/2-1,N)
x=xn*dx
yn=np.linspace(-N/2,N/2-1,N)
y=yn*dy
X, Y = np.meshgrid(x, y)
Z=np.empty([N, N], dtype='double')
ax = (1j * dt / hbar) * (hbar**2 / (mass * dx**2))*0.25
ay = (1j * dt / hbar) * (hbar**2 / (mass * dy**2))*0.25
Vy=0.5*wy**2*y**2*mass # 1D potential for y dimension
Vx=0.5*wx**2*x**2*mass # 1D potential for x dimension
# 1D wave functions for x dimension
absalphax=2.5
psicox=np.exp(-((x-x0*np.sqrt(2)*absalphax)/x0)**2/2)
normcox=np.sum(psicox*np.conjugate(psicox))*dx
psicox=psicox/np.sqrt(normcox)
# 1D wave functions for y dimension
psi0y=np.exp(-((y)/y0)**2/2)
norm0y=np.sum(psi0y*np.conjugate(psi0y))*dy
psi0y=psi0y/np.sqrt(norm0y)
psi1y=np.exp(-((y)/y0)**2/2)*2*y/y0
norm1y=np.sum(psi1y*np.conjugate(psi1y))*dy
psi1y=psi1y/np.sqrt(norm1y)
# Setting up 2D arrays for potential and wave function
V2D=np.empty([N, N], dtype=complex)
Psi2D=np.empty([N, N], dtype=complex)
a2D=np.empty([N, N], dtype=complex)
b2D=np.empty([N, N], dtype=complex)
for i in range(0,N):
    for j in range(0,N):
        V2D[i][j]=Vx[i]+Vy[j]
        Psi2D[i][j]=psicox[i]*1/np.sqrt(2)*(psi0y[j]+psi1y[j])
        a2D[i][j]=1.0 + 2.0 * ax + 2.0 * ay + 0.5*1j * dt / hbar * V2D[i][j]
        b2D[i][j]=1.0 - 2.0 * ax - 2.0 * ay - 0.5*1j * dt / hbar * V2D[i][j]
abx0=np.empty(N,dtype=object)
aby0=np.empty(N,dtype=object) 
for j in range(0,N):
       abx0[j]=np.sum(np.real(Psi2D[j,:]*np.conjugate(Psi2D[j,:])))*dy
for i in range(0,N):
       aby0[i]=np.sum(np.real(Psi2D[:,i]*np.conjugate(Psi2D[:,i])))*dx