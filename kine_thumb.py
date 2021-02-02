import sympy as sp
from sympy.physics.vector import init_vprinting
init_vprinting(use_latex='mathjax', pretty_print=False)
from IPython.display import Image
from sympy.physics.mechanics import dynamicsymbols
import math
from math import pi
import numpy as np
from sympy import *
alpha,d,theta,a,gamma,l_ck,l_pk,l_mk,l_dk,theta_1,theta_2,theta_3,theta_4,c,d,u=dynamicsymbols('alpha d theta a gamma l_ck l_pk,l_mk,l_dk,theta_1,theta_2,theta_3,theta_4,c,d,u')
#theta_1=dynamicsymbols('theta_1')
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
rot_1=sp.Matrix([[sp.cos(u),-sp.sin(u),0],
                 [sp.sin(u),sp.cos(u),0],
                 [0,0,1]])
trans_1=sp.Matrix([0,0,0])
m_05=sp.Matrix.vstack(sp.Matrix.hstack(rot_1, trans_1), last_row)
#m_08=m_05.subs({u:pi/6})
                 
#m01=m.subs({theta:pi})
#print(m_0)
m_1=m.subs({theta:sp.pi/2,d:0,alpha:theta_1,a:0})
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
m04=(m_0*m_1)
m=m04*m_03
m_1=m_05*m
#mbee1= sp.Matrix([m[0,0]],[m[1,0]],[m[2,0]],[1])
#print(sp.trigsimp(m04[0,0]))
#mbee12=sp.Matrix([[m04[0,0].simplify(), m04[0,1].simplify(), sp.trigsimp(m04[0,3].simplify())],[m04[1,0].simplify(), m04[1,1].simplify(), sp.trigsimp(m04[1,3].simplify())],[m04[2,0].simplify(), m04[2,1].simplify(), m04[2,2].simplify()]])
#print(m)
p_X=m_1[0,0]
p_Y=m_1[1,0]
p_Z=m_1[2,0]
print(p_X)
f_x=sp.lambdify((l_ck,l_pk,l_mk,l_dk,u,gamma,theta_1,theta_2,theta_3,theta_4),p_X,'numpy')
f_y=sp.lambdify((l_ck,l_pk,l_mk,l_dk,u,gamma,theta_1,theta_2,theta_3,theta_4),p_Y,'numpy')
f_z=sp.lambdify((l_ck,l_pk,l_mk,l_dk,u,gamma,theta_1,theta_2,theta_3,theta_4),p_Z,'numpy')
#print(f_x)
import matplotlib.pyplot as plt
#%matplotlib inline
import numpy as np
d2r = np.deg2rad
gamma=d2r(-40)
u=d2r(30)
theta_1=np.linspace(d2r(0), d2r(45))
theta_2=np.linspace(d2r(30),d2r(-15))
theta_3=np.linspace(d2r(0),d2r(-50))
theta_4=np.linspace(d2r(0),d2r(-80))
f_x=np.array(f_x(17,45,35,28,gamma,theta_1,theta_2,u,theta_3,theta_4))
#f_y=np.array(f_y(45,35,28,theta_1,theta_2,u,theta_3,theta_4))
f_y=np.array(f_y(17,45,35,28,u,gamma,theta_1,theta_2,theta_3,theta_4))
f_z=np.array(f_z(17,45,35,28,gamma,theta_1,theta_2,u,theta_3,theta_4))
##print(f_y)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

##<<solve for x, y, z here>>#

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(f_x,f_y,f_z)
plt.xlabel('x')
plt.ylabel('y')
#plt.zlabel('z')

plt.show()


