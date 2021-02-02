import sympy as sp
from sympy.physics.vector import init_vprinting
init_vprinting(use_latex='mathjax', pretty_print=False)
from IPython.display import Image
from sympy.physics.mechanics import dynamicsymbols
import math
from math import pi
import numpy as np
import scipy
import scipy.linalg
from mat_mul import mat_mul1
#import scipy.linalg as spla
alpha,d,theta,a,gamma,l_ck,l_pk,l_mk,l_dk,theta_01,theta_1,theta_2,theta_3,theta_4,c,d=dynamicsymbols('alpha d theta a gamma l_ck l_pk,l_mk,l_dk,theta_01,theta_1,theta_2,theta_3,theta_4,c,d')
theta_1=dynamicsymbols('theta_1')
#print(alpha,d,theta,a)
rot = sp.Matrix([[sp.cos(alpha), -sp.sin(alpha)*sp.cos(theta), sp.sin(alpha)*sp.sin(theta)],
                 [sp.sin(alpha), sp.cos(alpha)*sp.cos(theta), -sp.cos(alpha)*sp.sin(theta)],
                 [0, sp.sin(theta), sp.cos(theta)]])

trans = sp.Matrix([a*sp.cos(alpha),a*sp.sin(alpha),0])
print(trans.shape)

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
f_x=sp.lambdify((l_ck,l_pk,l_mk,l_dk,gamma,theta_1,theta_2,theta_3,theta_4),p_X,'numpy')
f_y=sp.lambdify((l_ck,l_pk,l_mk,l_dk,gamma,theta_1,theta_2,theta_3,theta_4),p_Y,'numpy')
f_z=sp.lambdify((l_ck,l_pk,l_mk,l_dk,gamma,theta_1,theta_2,theta_3,theta_4),p_Z,'numpy')
#(p_Z)
from autograd import grad, jacobian
m_fj=sp.Matrix([[p_X],[p_Y],[p_Z]])
param=([theta_1,theta_2,theta_3,theta_4])
l_1=m_fj.jacobian(param)
l_01=sp.lambdify((l_ck,l_pk,l_mk,l_dk,gamma,theta_1,theta_2,theta_3,theta_4),l_1,'numpy')
import numpy as np
import random
from numpy import *
d2r = np.deg2rad
dr2=np.rad2deg
gamma=d2r(0)
theta_1=d2r(0)
theta_2=d2r(0)
theta_3=d2r(0)
theta_4=d2r(40)
deltheta=np.zeros((4,1))

l=np.zeros((3,1))
l_ck=85
l_pk=43
l_mk=25
l_dk=17
l=np.zeros((3,1))
l_m=np.zeros((3,1))
l_m[0,0]=125.85618841
l_m[1,0]=62.32422961
l_m[2,0]= 26.45643137
#l_m=[[5.91743855e+01],[3.33671192e+01,2],[49975503e+01]]
jaco=np.zeros((3,4))
y=np.zeros((3,1))
f=np.zeros((3,3))
f_1=np.zeros((3,3))
f_11=np.zeros((4,3))
err=np.zeros((3,1))
#print(l_m)
k=0
while (k<1000):
        
    #try:
        l[0,0]=(np.array(f_x(l_ck,l_pk,l_mk,l_dk,gamma,theta_1,theta_2,theta_3,theta_4)))
        l[1,0]=(np.array(f_y(l_ck,l_pk,l_mk,l_dk,gamma,theta_1,theta_2,theta_3,theta_4)))
        l[2,0]=(np.array(f_z(l_ck,l_pk,l_mk,l_dk,gamma,theta_1,theta_2,theta_3,theta_4)))
        err[0,0]=(l_m[0,0]-l[0,0])
        err[1,0]=(l_m[1,0]-l[1,0])
        err[2,0]=(l_m[2,0]-l[2,0])
        jaco=np.array(l_01(l_ck,l_pk,l_mk,l_dk,gamma,theta_1,theta_2,theta_3,theta_4))
        f_1=mat_mul1((jaco),np.transpose(jaco),3,3)
        u,sigma,u_trans=scipy.linalg.svd(f_1,full_matrices=True)
        for j in range((len(m_fj))):
                if sigma[j]<=0.0001:
                        sigma[j]=sigma[j]+1
                else:
                        sigma[j]=sigma[j]
        for i in range((len(m_fj))):
                f[i,i]=1/sigma[j]
        y=mat_mul1(np.transpose(u_trans),f,3,3)
        y_1=mat_mul1(y,u,3,3)
        
        f_11=mat_mul1(np.transpose(jaco),y_1,4,3)
        deltheta=mat_mul1(f_11,err,4,1)
        theta_1=(theta_1+deltheta[0,0])%(2*pi)
        theta_2=(theta_2+deltheta[1,0])%(2*pi)
        theta_3=(theta_3+deltheta[2,0])%(2*pi)
        theta_4=(theta_4+deltheta[3,0])%(2*pi)
        k=k+1
        
print(dr2(theta_1),dr2(theta_2),dr2(theta_3),dr2(theta_4))
                
        
               # for i in range((len(m_fj))):
                    #f[i,i]=sigma[j]
       # print(f)
        #y=mat_mul1(np.transpose(u_trans),f,3,3)
        #y_1=mat_mul1(y,u,3,3)
        #f_inv=np.linalg.inv[Y_1]
        #f_11=mat_mul1(np.transpose(jaco),f_inv,4,3)
        #resi=(l_m-1)
        ##deltheta=mat_mul1(f_l1,resi,4,1)
        #theta_1=theta_1+deltheta[0,0]
        #theta_2=theta_2+deltheta[1,0]
        #theta_3=theta_3+deltheta[2,0]
        #theta_4=theta_4+deltheta[3,0]
            #print(theta_1)
    #except :
        #print("no sl exist")
    #k=k+1
        #print(theta_3)
    #else:
         #theta_1=theta_1
         #theta_2=theta_2
         #theta_3=theta_3
         #theta_4=theta_4
    



#print(theta_1)
#print(theta_2)
#print(theta_3)
#print(theta_4)

#print(l_1.shape)
