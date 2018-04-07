
# coding: utf-8

# In[54]:


from pyquil.quil import Program
from pyquil.api import QVMConnection
from pyquil.gates import *
from pyquil.parameters import Parameter, quil_sin, quil_cos
from pyquil.quilbase import DefGate
import numpy as np


# Initialize $R_y^{\theta_1}$ 

# In[66]:


theta = Parameter('theta')

ccry = np.eye(8, dtype = object)
ccry[6][6] = quil_cos(theta / 2)
ccry[6][7] = - quil_sin(theta / 2)
ccry[7][6] = quil_sin(theta / 2)
ccry[7][7] = quil_cos(theta / 2)

dg0 = DefGate('CCRY', ccry, [theta])
CCRY = dg0.get_constructor()

print (ccry)


# Initialize $R_y^{\theta_2}$ 

# In[67]:


cncry = np.eye(8, dtype = object)
cncry[2][2] = quil_cos(theta / 2)
cncry[2][3] = - quil_sin(theta / 2)
cncry[3][2] = quil_sin(theta / 2)
cncry[3][3] = quil_cos(theta / 2)

dg1 = DefGate('CNCRY', cncry, [theta])
CNCRY = dg1.get_constructor()


# Testing 

# In[68]:


p = Program()   # clear the old program
p.inst(dg0, X(0), X(1), X(2), CCRY(np.pi)(0, 1, 2))

qvm = QVMConnection()
wavefunction = qvm.wavefunction(p)

print(wavefunction)


# In[69]:


p = Program()   # clear the old program
p.inst(dg1, X(0), X(1), X(2), CNCRY(np.pi)(0, 1, 2))

qvm = QVMConnection()
wavefunction = qvm.wavefunction(p)

print(wavefunction)


# In[70]:


theta_1 = 0
theta_2 = 0

p = Program()   # whatever they give us
p.inst(CNCRY(theta_2)(3, 2, 1))
p.inst(CCRY(theta_1)(3, 2, 1))

