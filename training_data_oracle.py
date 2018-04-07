from pyquil.quil import Program
from pyquil.api import QVMConnection
from pyquil.gates import *
from pyquil.parameters import Parameter, quil_sin, quil_cos
from pyquil.quilbase import DefGate

from pyquil.parameters import Parameter, quil_sin, quil_cos
from pyquil.quilbase import DefGate
import numpy as np

import pyquil

from pyquil.quil import Program
from pyquil.gates import X,Z,Y
from pyquil.gates import MEASURE
from pyquil.gates import H
from pyquil.gates import CNOT
from pyquil.api import QVMConnection
from pyquil.gates import SWAP

from image_processing import *

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

def prog_oracle(p, theta_1, theta_2):
    p.inst(dg1, CNCRY(theta_2)(3, 2, 1))
    p.inst(dg0, CCRY(theta_1)(3, 2, 1))
    return p

def matrix_inversion():
    pass

def prog_u(prog, vec_querry=None):
    if(vec_querry==None):
        x0 = [0.997, -0.072] # paper first 6 querry
    else:
        x0 = vec_querry

    theta_zero = get_theta(x0)

    dg1, ch = get_cH()
    dg2, cry = get_cRy()

    prog += dg1
    prog += dg2
    prog += cry(-theta_zero)(3, 1)
    prog += ch(3, 1)
    prog += SWAP(0, 3)

    return prog


def get_theta(x_i):
    # ArcCot[z] is equal to ArcTan[1/z] for complex z, so also R
    theta = np.arctan(x_i[0]/x_i[1])
    return theta

def get_F(gamma=2, kernel_matrix=None):
    # for n=2

    # k
    k = [[0.5065, 0.2425],[0.2425, 0.4935]]
    i = [[1, 0],[0, 1]]
    f = k + 1/gamma * i

    return f

# 'black circle' case (first qubit supplied is 1)
def get_cRy():
    theta = Parameter('theta')
    #crx = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, quil_cos(theta / 2), -1j * quil_sin(theta / 2)], [0, 0, -1j * quil_sin(theta / 2), quil_cos(theta / 2)]])
    cry = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, quil_cos(theta / 2), -1 * quil_sin(theta / 2)], [0, 0, +1 * quil_sin(theta / 2), quil_cos(theta / 2)]])
    
    dg = DefGate('CRY', cry, [theta])
    CRY = dg.get_constructor()

    return [dg, CRY]

def get_cH():
    #dummy = Parameter('dummy')
    ch = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1/np.sqrt(2), 1/np.sqrt(2)], [0, 0, 1/np.sqrt(2), -1/np.sqrt(2)]])
    
    dg = DefGate('CH', ch)
    Ch = dg.get_constructor()

    return [dg, Ch]


def main():
    qvm = QVMConnection()
    prog = Program()


    path = "6.png"
    img_6 = load_image(path)
    query = full_process(img_6)
    print("Features (HR, VR) for", path, query)
   

    prog = prog_oracle(prog, 1, 2) 
    prog = prog_u(prog)
    print(prog)

    results = qvm.run(prog, classical_addresses =[0,1], trials=10)
    # requires access
    #q_results = qpu.run(prog2, classical_addresses =[0,1], trials=10)

    print(results)

main()
