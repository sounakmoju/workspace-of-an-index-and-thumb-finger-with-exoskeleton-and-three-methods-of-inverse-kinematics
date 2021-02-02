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
import random
import numpy as np
import pandas as pd
from pylab import plot,show
from numpy import vstack,array
from numpy.random import rand
from scipy.cluster.vq import kmeans,vq
from sklearn.cluster import KMeans
import kmeans1d
from math import sqrt
from math import pi
from math import exp
import numpy as np
import pandas as pd
from naive_bias import class_separation,mean,std_dev,summarize,summarize_cls,calculate_probability
import pandas as pd
import random
l_m=np.zeros((3,1))
l_m[0,0]=69.60158339 
l_m[1,0]=53.29799633 
l_m[2,0]= 22.6235733
df1=pd.read_csv("kine_features1.csv")
print(df1.head())
data_12=df1.values
norm=summarize_cls(data_12)
#print(data_12[1])
#print(class_separation(data_12))


d=list()
df=pd.read_csv("kine_class_1.csv")
#print(df.head())
data_1=df["dist"]
data_2=df.values
##creating clusters from worspace for sampling the theta_initial
clusters, centroids = kmeans1d.cluster(data_1,5)
# assign each sample to a cluster
#km.fit(x.reshape(-1,1))
for i in centroids:
    d.append(i)
##def Reverse(Ist): 
    #Ist.reverse()
    #return Ist
      
# Driver Code 
#lst = [10, 11, 12, 13, 14, 15] 
#print(Reverse(d))
d_1=d[::-1]
f1={0:d_1[0],1:d_1[1],2:d_1[2],3:d_1[3],4:d_1[4]}
l_m_1=sqrt(pow(l_m[0,0],2)+pow(l_m[1,0],2)+pow(l_m[2,0],2))
d2=[]

def det_mini(f1,l_m_1):
    for key in f1:
        di=pow((f1[key]-l_m_1),2)
        d2.append(di)
    min_index=d2.index(min(d2))
    return min_index
mini_1=det_mini(f1,l_m_1)
theta=[]
def sam_probablity(mean,std):
    s=np.random.normal(mean,std,1000)
    p1=[]
    for i in s:
        p1.append(calculate_probability(i,mean,std))
    idx = np.argmax(p1)
    k=idx
    p=s[k]
    #print(len(p1))
        
    #if p1<0.75:
        #sam_probablity(mean,std)
        
    
    
    #else:
        #p=s
    return p
    
for i in norm[mini_1]:
    mean,std=i
    k=sam_probablity(mean,std)
    theta.append(k)
theta12=theta
da_coeff=0.0000001
#import scipy.linalg as spla
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
gamma=d2r(-15)
theta_1=(d2r(theta12[0]))
theta_2=d2r(theta12[1])
theta_3=d2r(theta12[2])
theta_4=d2r(theta12[3])
#-6.63265306  22.55102041
  # 26.53061224  21.2244898
#-20.40816327  69.3877551
   #81.63265306  65.30612245




#print(theta_1)

deltheta=np.zeros((4,1))

l=np.zeros((3,1))
l_ck=85
l_pk=43
l_mk=25
l_dk=17
l=np.zeros((3,1))


#69.60158339  53.29799633  22.6235733
#59.11339277  27.28741052  25.63253116
#164.25515487   4.55011208  43.22982229
#l_m=[[5.91743855e+01],[3.33671192e+01,2],[49975503e+01]]
jaco=np.zeros((3,4))
y=np.zeros((3,1))
f=np.zeros((3,3))
f_1=np.zeros((3,3))
f_11=np.zeros((4,3))
err=np.zeros((3,1))
d_0=np.zeros((3,3))
#print(l_m)
k=0
while (k<50):
        l[0,0]=(np.array(f_x(l_ck,l_pk,l_mk,l_dk,gamma,theta_1,theta_2,theta_3,theta_4)))
        l[1,0]=(np.array(f_y(l_ck,l_pk,l_mk,l_dk,gamma,theta_1,theta_2,theta_3,theta_4)))
        l[2,0]=(np.array(f_z(l_ck,l_pk,l_mk,l_dk,gamma,theta_1,theta_2,theta_3,theta_4)))
        err[0,0]=(l_m[0,0]-l[0,0])
        err[1,0]=(l_m[1,0]-l[1,0])
        err[2,0]=(l_m[2,0]-l[2,0])
        jaco=np.array(l_01(l_ck,l_pk,l_mk,l_dk,gamma,theta_1,theta_2,theta_3,theta_4))
        u,d,v_t=scipy.linalg.svd(jaco,full_matrices=True)
        
        #print(jaco)
        d_0[0,0]=d[0]
        d_0[1,1]=d[1]
        d_0[2,2]=d[2]
        d_1=mat_mul1(d_0,d_0,3,3)
       
        d_12= (da_coeff*(np.eye(3)))
        sigma=scipy.linalg.inv((d_1+d_12))
        #print(d_1)
        g_1=mat_mul1(d_0,sigma,4,3)
        #print(g_1)
        g_2=mat_mul1(v_t.transpose(),g_1,4,3)
        #print(g_2)
        g_3=mat_mul1(g_2,u.transpose(),4,3)#print(g_3)
        del_theta=mat_mul1(g_3,err,4,1)
        #print(del_theta)
        theta_1=(theta_1+del_theta[0,0])
        theta_2=(theta_2+del_theta[1,0])
        theta_3=(theta_3+del_theta[2,0])
        theta_4=(theta_4+del_theta[3,0])
        #print(theta_1)
        
        #print(deltheta)

        k=k+1


print(dr2(theta_1),dr2(theta_2),dr2(theta_3),dr2(theta_4))
print((theta_1),(theta_2),(theta_3),(theta_4))
print((err[0,0],err[1,0],err[2,0]))
