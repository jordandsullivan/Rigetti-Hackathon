from pyquil.quil import Program
from pyquil.api import QVMConnection
from pyquil.gates import *
from pyquil.parameters import Parameter, quil_sin, quil_cos
from pyquil.quilbase import DefGate
import numpy as np

theta = Parameter('theta')

ccry = np.eye(8, dtype = object)
ccry[6][6] = quil_cos(theta / 2)
ccry[6][7] = - quil_sin(theta / 2)
ccry[7][6] = quil_sin(theta / 2)
ccry[7][7] = quil_cos(theta / 2)

dg0 = DefGate('CCRY', ccry, [theta])
CCRY = dg0.get_constructor()

cncry = np.eye(8, dtype = object)
cncry[2][2] = quil_cos(theta / 2)
cncry[2][3] = - quil_sin(theta / 2)
cncry[3][2] = quil_sin(theta / 2)
cncry[3][3] = quil_cos(theta / 2)

dg1 = DefGate('CNCRY', cncry, [theta])
CNCRY = dg1.get_constructor()

def test_CCRY():
    p = Program()   # clear the old program
    p.inst(X(0), X(1), X(2), dg0, CCRY(np.pi)(0, 1, 2))

    qvm = QVMConnection()
    wavefunction = qvm.wavefunction(p)

    print(wavefunction)

def test_CNCRY():
    p = Program()   # clear the old program
    p.inst(X(0), X(1), X(2), dg1, CNCRY(np.pi)(0, 1, 2))

    qvm = QVMConnection()
    wavefunction = qvm.wavefunction(p)

    print(wavefunction)

def add_training_oracle(p, theta_1, theta_2):
    p.inst(dg1, CNCRY(theta_2)(3, 2, 1))
    p.inst(dg0, CCRY(theta_1)(3, 2, 1))
    return p

