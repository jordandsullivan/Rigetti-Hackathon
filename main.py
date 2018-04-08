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

import numpy as np
import scipy
from scipy.linalg import expm, sinm, cosm

from image_processing import *

theta = Parameter('theta')

K = np.array([[0.5065, 0.2425], 
              [0.2425, 0.4935]])
gamma = 1000 #2
F = K + 1/gamma*np.eye(2)


class Inverse():
    def __init__(self, F, program, qvm, theta0, theta1, theta2):
        self.F = F
        self.program = program
        self.qvm = qvm
        self.first, self.second, self.y, self.anc = (0,1,2,3)
        
        self.theta0 = theta0
        self.theta1 = theta1
        self.theta2 = theta2

#     def expmatrix(self, val,matrix):  
#         return expm(val*1j*self.F)
    
    def controlMake(self, M):
        """Creates control gates from M.
        param:
            M: (matrix) to control
        returns:
            (matrix) controlled."""
        zero = np.zeros((2,2))
        I = np.eye(2)
        top = np.concatenate((I, zero), axis = 1)
        bottom = np.concatenate((zero, M), axis = 1)
        res = np.concatenate((top, bottom), axis = 0)
        return res
    
    def CRY(self, angle):
        """Creates control RY gate.
        param:
            angle: (float) by how much to rotate
        returns:
            matrix."""
        Y = np.array([[0, -1j], [1j, 0]])
        expY = expm(-angle/2. * 1j * Y)
        return self.controlMake(expY)
    
    def CCRY(self, angle):
        Y = np.array([[0, -1j], [1j, 0]])
        expY = expm(-angle/2. * 1j * Y)
        
        ccry = np.eye(8,dtype='cfloat')
        ccry[6][6] = expY[0][0]
        ccry[6][7] = expY[0][1]
        ccry[7][6] = expY[1][0]
        ccry[7][7] = expY[1][1]

        return ccry
    
    def NCCRY(self, angle):
        Y = np.array([[0, -1j], [1j, 0]])
        expY = expm(-angle/2. * 1j * Y)
        
        cncry = np.eye(8,dtype='cfloat')
        cncry[2][2] = expY[0][0]
        cncry[2][3] = expY[0][1]
        cncry[3][2] = expY[1][0]
        cncry[3][3] = expY[1][1]
        
        return cncry
        
    def HGate(self):
        return self.controlMake(np.sqrt(0.5) * np.array([[1,1], [1,-1]]))
    
    
    def run(self):

        # add hadamards
        self.program += H(self.first)
        self.program += H(self.second)
        
        # add the exponent gates
        expF = expm(np.pi*1j*self.F)
        expF = self.controlMake(expF)
        self.program = self.program.defgate("expF", expF)
        self.program.inst(("expF", self.first, self.y))
        
        expFhalf = expm(np.pi/2.*1j*self.F)
        expFhalf = self.controlMake(expFhalf)
        self.program = self.program.defgate("expFhalf", expFhalf)
        self.program.inst(("expFhalf", self.second, self.y))
        
        self.program += SWAP(self.first, self.second)
        self.program += H(self.second)
        
        #S inverse
        self.program += CPHASE(-np.pi/2, self.second, self.first) # right order of qubits?
        
        self.program += H(self.first)
        
        CRYpi4 = self.CRY(np.pi/4.)
        self.program = self.program.defgate("CRYpi4", CRYpi4)
        self.program.inst(("CRYpi4", self.second, self.anc))
        
        CRYpi8 = self.CRY(np.pi/8.)
        self.program = self.program.defgate("CRYpi8", CRYpi8)
        self.program.inst(("CRYpi8", self.first, self.anc))  
        
        self.program += H(self.first)
        
        self.program += CPHASE(np.pi/2, self.second, self.first) # right order of qubits?
        
        self.program += H(self.second)
        
        self.program += SWAP(self.first, self.second)
        
        minusExpFhalf = expm(-np.pi/2.*1j*self.F)
        minusExpFhalf = self.controlMake(minusExpFhalf)
        self.program = self.program.defgate("minusExpFhalf", minusExpFhalf)
        self.program.inst(("minusExpFhalf", self.second, self.y))
        
        minusExpF = expm(-np.pi*1j*self.F)
        minusExpF = self.controlMake(minusExpF)
        self.program = self.program.defgate("minusExpF", minusExpF)
        self.program.inst(("minusExpF", self.second, self.y))
        
        self.program += H(self.first)
        self.program += H(self.second)
        
        theta1Gate = self.NCCRY(self.theta1)
        self.program = self.program.defgate("theta1Gate", theta1Gate)
        self.program.inst(("theta1Gate", self.anc, self.y, self.second))
        
        theta2Gate = self.CCRY(self.theta2)
        self.program = self.program.defgate("theta2Gate", theta2Gate)
        self.program.inst(("theta2Gate", self.anc, self.y, self.second))
        
        theta0Gate = self.CRY(-self.theta0)
        self.program = self.program.defgate("theta0Gate", theta0Gate)
        self.program.inst(("theta0Gate", self.anc, self.second))
        
        CH = self.HGate()
        self.program = self.program.defgate("CH", CH)
        self.program.inst(("CH", self.anc, self.y))
        
        self.program += SWAP(self.first, self.anc)
        
        self.program += MEASURE(self.anc, [0]) # or 1?


        wavefunction = (self.qvm.wavefunction(self.program))
        ampl = wavefunction.amplitudes
        #ancil = self.qvm.run(self.program, classical_addresses=[0], trials=1)
        
        if(np.real(ampl[0]) == 0):
            print("Wavefunction:", wavefunction)
            #print("real ", np.real(ampl[0]))
            return wavefunction
        else:
            return None
        #result = self.qvm.run(self.program, classical_addresses=[0], trials=10)

def main():
    # load data
    path_6 = "6.png"
    img_6 = load_image(path_6)
    path_9 = "9.png"
    img_9 = load_image(path_9)
    query_0 = full_process(img_6)
    query_1 = full_process(img_9)
    n_correct = 0
    n_runs = 30

    for i in range(n_runs):
        print("Features (HR, VR) for ", path_6)
        print(query_0)
        if( query(query_0) == 6):
            n_correct += 1
        print(str(i) + " Classified as", query(query_0))
        print("Features (HR, VR) for ", path_9)
        print(query_1)
        if( query(query_1) == 9):
            n_correct += 1
        print(str(i) + "Classified as", query(query_1))

        print("So far " + str(n_correct) + "/" + str(2 * (i+1)))

    print("Classified correctly ", n_correct, "out of ", 2*i)

def calc_theta(x_i):
    # ArcCot[z] is equal to ArcTan[1/z] for complex z, so also R
    theta = np.arctan(1/(x_i[0]/x_i[1]))
    return theta

def query(q):

    train6 = [0.987, 0.159]
    train9 = [0.354, 0.935]
    theta1 = calc_theta(train6)
    theta2 = calc_theta(train9)

   
    theta0 = calc_theta(q)

    inverse = Inverse(F, Program(), QVMConnection(), theta0, theta1, theta2)
    
    amplitudes = None
    while(amplitudes == None):
        amplitudes = inverse.run()

    

    if(np.real(amplitudes[1]) > 0):
        return 6
    else:
        return 9

main()
        



        
        

        
        
        


        
