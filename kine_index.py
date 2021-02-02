import sympy as sp
from sympy.physics.vector import init_vprinting
init_vprinting(use_latex='mathjax', pretty_print=False)
from IPython.display import Image
from sympy.physics.mechanics import dynamicsymbols
import math
from math import pi
import numpy as np
alpha,d,theta,a,gamma,l_ck,l_pk,l_mk,l_dk,theta_01,theta_1,theta_2,theta_3,theta_4,c,d=dynamicsymbols('alpha d theta a gamma l_ck l_pk,l_mk,l_dk,theta_01,theta_1,theta_2,theta_3,theta_4,c,d')
theta_1=dynamicsymbols('theta_1')
#print(alpha,d,theta,a)
rot = sp.Matrix([[sp.cos(alpha), -sp.sin(alpha)*sp.cos(theta), sp.sin(alpha)*sp.sin(theta)],
                 [sp.sin(alpha), sp.cos(alpha)*sp.cos(theta), -sp.cos(alpha)*sp.sin(theta)],
                 [0, sp.sin(theta), sp.cos(theta)]])

trans = sp.Matrix([a*sp.cos(alpha),a*sp.sin(alpha),0])

last_row = sp.Matrix([[0, 0, 0, 1]])

m = sp.Matrix.vstack(sp.Matrix.hstack(rot, trans), last_row)
rot1=sp.Matrix([[sp.cos(gamma),0,sp.sin(gamma)],
                [0,1,0],
                [-sp.sin(gamma),0,sp.cos(gamma)]])
trans=sp.Matrix([l_ck*sp.cos(gamma),0,-l_ck*sp.sin(gamma)])
#last_row=sp.Matrix([[0, 0, 0, 1]])
m_0=sp.Matrix.vstack(sp.Matrix.hstack(rot1, trans), last_row)
#m01=m.subs({theta:pi})
#print(m_0)
m_01=m.subs({theta:sp.pi/2,d:0,alpha:0,a:0})
m_1=m.subs({theta:-sp.pi/2,d:0,alpha:theta_1,a:0})
m_2=m.subs({theta:0,d:0,alpha:theta_2,a:l_pk})
m_3=m.subs({theta:0,d:0,alpha:theta_3,a:l_mk})
m_4=m.subs({theta:0,d:0,alpha:theta_4,a:l_dk})
m02 = (m_2*m_3*m_4)
mbee= sp.Matrix([[m02[0,0].simplify(), m02[0,1].simplify(), sp.trigsimp(m02[0,3].simplify())],
                 [m02[1,0].simplify(), m02[1,1].simplify(), sp.trigsimp(m02[1,3].simplify())],
                 [m02[2,0].simplify(), m02[2,1].simplify(), m02[2,2].simplify()]])

c=mbee[0,2]
d=mbee[1,2]
#m03=sp.Matrix([[mbee[0,2]],[mbee[1,2]],[0],[1]])
m_03=sp.Matrix([[c],[d],[0],[1]])
m04=(m_0*m_01*m_1)
m=m04*m_03
#mbee1= sp.Matrix([m[0,0]],[m[1,0]],[m[2,0]],[1])
#print(sp.trigsimp(m04[0,0]))
#mbee12=sp.Matrix([[m04[0,0].simplify(), m04[0,1].simplify(), sp.trigsimp(m04[0,3].simplify())],[m04[1,0].simplify(), m04[1,1].simplify(), sp.trigsimp(m04[1,3].simplify())],[m04[2,0].simplify(), m04[2,1].simplify(), m04[2,2].simplify()]])
m_12=sp.Matrix([[sp.trigsimp(m[0,0].simplify())],[m[1,0]],[sp.trigsimp(m[2,0].simplify())],[1]])
p_X=m_12[0,0]
p_Y=m_12[1,0]
p_Z=m_12[2,0]
##print(p_Z)
f_x=sp.lambdify((l_ck,l_pk,l_mk,l_dk,gamma,theta_1,theta_2,theta_3,theta_4),p_X,'numpy')
f_y=sp.lambdify((l_ck,l_pk,l_mk,l_dk,gamma,theta_1,theta_2,theta_3,theta_4),p_Y,'numpy')
f_z=sp.lambdify((l_ck,l_pk,l_mk,l_dk,gamma,theta_1,theta_2,theta_3,theta_4),p_Z,'numpy')
#print(f_x)
import matplotlib.pyplot as plt
#%matplotlib inline
import numpy as np
d2r = np.deg2rad
gamma=d2r(-15)
u=d2r(30)
theta_1=np.linspace(d2r(0), d2r(-25))
theta_2=np.linspace(d2r(0),d2r(85))
theta_3=np.linspace(d2r(0),d2r(100))
theta_4=np.linspace(d2r(0),d2r(80))
#theta_1=-0.12217304763960307 
#theta_2=(0.04642647698824287)
#theta_3=(0.8595876904229619)
#theta_4=(0.1669278086770553)

f_x=np.array(f_x(85,43,25,17,gamma,theta_1,theta_2,theta_3,theta_4))
#f_y=np.array(f_y(45,35,28,theta_1,theta_2,u,theta_3,theta_4))
f_y=np.array(f_y(85,43,25,17,gamma,theta_1,theta_2,theta_3,theta_4))
f_z=np.array(f_z(85,43,25,17,gamma,theta_1,theta_2,theta_3,theta_4))
#print(f_x,f_y,f_z)
m_1=len(f_x)
M=np.zeros((m_1,7))
dr2=np.rad2deg
d2r=np.deg2rad

#M=np.zeros((1,7))
#M_1=np.zeros((1,3))
M[:,0]=f_x
M[:,1]=f_y
M[:,2]=f_z
M[:,3]=dr2(theta_1)
M[:,4]=dr2(theta_2)
M[:,5]=dr2(theta_3)
M[:,6]=dr2(theta_4)
#np.savetxt("kine_class.csv", M, delimiter=",")
#k_1=np.array(f_x(85,43,25,17,gamma,-0.15707963267948966, 0.84008051400167,1.53588974175501,1.239183768915974))
##M_1[0,1]=np.array(f_y(85,43,25,17,gamma,-0.15707963267948966,0.84008051400167,1.53588974175501,1.239183768915974))
#M_1[0,2]=np.array(f_z(85,43,25,17,gamma,-0.15707963267948966, 0.84008051400167, 1.53588974175501, 1.239183768915974))


print(M)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

##<<solve for x, y, z here>>#

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(-f_x,-f_y,-f_z)
plt.xlabel('x')
plt.ylabel('y')
#plt.zlabel('z')

plt.show()
#print(m_12[3,0])
