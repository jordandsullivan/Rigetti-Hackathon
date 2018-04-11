import pyquil
from pyquil.quil import Program
from pyquil.api import QVMConnection
from pyquil.gates import *
from pyquil.parameters import Parameter, quil_sin, quil_cos
from pyquil.quilbase import DefGate
from pyquil.quil import Program
from pyquil.gates import X,Z,Y
from pyquil.gates import MEASURE
from pyquil.gates import H
from pyquil.gates import CNOT
from pyquil.api import QVMConnection
from pyquil.gates import SWAP
from pyquil.parameters import Parameter, quil_sin, quil_cos
from pyquil.quilbase import DefGate

import numpy as np
import scipy
from scipy.linalg import expm, sinm, cosm

from image_processing import *

theta = Parameter('theta')

K = np.array([[0.5065, 0.2425], 
              [0.2425, 0.4935]])
gamma = 100 # 2
F = K + 1/gamma*np.eye(2)


class QSVM():
    def __init__(self, F, qvm):
        self.F = F
        self.program = Program()    # empty pyquil program
        self.qvm = qvm
        self.first, self.second, self.y, self.anc = (0,1,2,3)
        
        self.theta0 = None
        self.theta1 = None
        self.theta2 = None

        self.verb_print_program = False
        self.verb_print_wavefunction = False


    def train(self, vec_train_0, vec_train_1):
        """
        @brief Train the SVM
        @param vec_train_0: training vector 0. Paper: 6
        @param vec_train_0: training vector 1. Paper: 9
        """
        theta1 = calc_theta(vec_train_0)
        theta2 = calc_theta(vec_train_1)
        self.theta1 = theta1
        self.theta2 = theta2

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
    
    def get_program_inversion(self, program):
        # =============================================
        # Inversion 
        # =============================================
        # add hadamards
        program += H(self.first)
        program += H(self.second)

        # add the exponent gates
        expF = expm(np.pi*1j*self.F)
        expF = self.controlMake(expF)
        program = program.defgate("expF", expF)
        program.inst(("expF", self.first, self.y))

        expFhalf = expm(np.pi/2.*1j*self.F)
        expFhalf = self.controlMake(expFhalf)
        program = program.defgate("expFhalf", expFhalf)
        program.inst(("expFhalf", self.second, self.y))

        program += SWAP(self.first, self.second)
        program += H(self.second)

        #S inverse
        program += CPHASE(-np.pi/2, self.second, self.first) # right order of qubits?

        program += H(self.first)

        CRYpi4 = self.CRY(np.pi/4.)
        program = program.defgate("CRYpi4", CRYpi4)
        program.inst(("CRYpi4", self.second, self.anc))

        CRYpi8 = self.CRY(np.pi/8.)
        program = program.defgate("CRYpi8", CRYpi8)
        program.inst(("CRYpi8", self.first, self.anc))  

        program += H(self.first)

        program += CPHASE(np.pi/2, self.second, self.first) # right order of qubits?

        program += H(self.second)

        program += SWAP(self.first, self.second)

        minusExpFhalf = expm(-np.pi/2.*1j*self.F)
        minusExpFhalf = self.controlMake(minusExpFhalf)
        program = program.defgate("minusExpFhalf", minusExpFhalf)
        program.inst(("minusExpFhalf", self.second, self.y))

        minusExpF = expm(-np.pi*1j*self.F)
        minusExpF = self.controlMake(minusExpF)
        program = program.defgate("minusExpF", minusExpF)
        program.inst(("minusExpF", self.second, self.y))

        program += H(self.first)
        program += H(self.second)

        return program

    def get_program_oracle(self, program):
        # =============================================
        # Training Data Orcale 
        # =============================================

        theta1Gate = self.NCCRY(self.theta1)
        program = program.defgate("theta1Gate", theta1Gate)
        program.inst(("theta1Gate", self.anc, self.y, self.second))

        theta2Gate = self.CCRY(self.theta2)
        program = program.defgate("theta2Gate", theta2Gate)
        program.inst(("theta2Gate", self.anc, self.y, self.second))
        
        return program 
        
    def get_program_u(self, program):

        # =============================================
        # U_x0
        # =============================================

        theta0Gate = self.CRY(-self.theta0)
        program = program.defgate("theta0Gate", theta0Gate)
        program.inst(("theta0Gate", self.anc, self.second))

        CH = self.HGate()
        program = program.defgate("CH", CH)
        program.inst(("CH", self.anc, self.y))

        return program


    def run_quantum(self, vec_query, mes_anc=True, mes_rep=True):
        """
        @brief Run the quantum SVM to classify a given query
        @param vec_query: query vector. 
        @return wavefunction after run
        """
        
        self.theta0 = calc_theta(vec_query)
        
        self.program = self.get_program_inversion(self.program)
        self.program = self.get_program_oracle(self.program)
        self.program = self.get_program_u(self.program)


        # =============================================
        # MEASUREMENT
        # =============================================

        self.program += SWAP(self.first, self.anc)
        if mes_anc:
            self.program += MEASURE(self.anc, 0)

        if self.verb_print_program:
            print("Pyquil program: \n", self.program)

        # only return, if ancilla measured 1 (a little hacky by amplitues)
        while True:
            wavefunction = (self.qvm.wavefunction(self.program))

            if mes_rep:
                # repetitive, conditional readout like on hackathon
                if np.real(wavefunction.amplitudes[0]) == 0 and np.imag(wavefunction.amplitudes[0]) == 0:
                    if self.verb_print_wavefunction:
                        print("Final Wavefunction: " + str(wavefunction))
                        print("Ampl[9] (|1001>): " + str(wavefunction.amplitudes[9]))
                    return wavefunction

            else:
                if self.verb_print_wavefunction:
                    print(wavefunction)
                return wavefunction

            
            
            # not the same as checking amplitude! only statistical that ancilla is 0!
            #results = qvm.run(program, classical_addresses =[0], trials=1)
            #if(results[0][0] == 1):
            #    print("wavefunction" + str(wavefunction))
            #    print("ampl[9] (|1001>)" + str(wavefunction.amplitudes[9]))
            #    return wavefunction

        
def main():    
    path_6 = "6.png"
    img_6 = load_image(path_6)
    path_9 = "9.png"
    img_9 = load_image(path_9)
    img_list_6 = [img_6]
    img_list_9 = [img_9]
    n_correct = 0
    n_correct_t1 = 0
    n_rep = 5
    
    # first index is correct class
    # all examples from paper
    query_list = [[6, [0.997,-0.072]], [9, [0.147,0.989]], [6, [0.999,-0.030]], [6, [0.987,-0.161]],
                 [9, [0.338,0.941]], [6, [0.999,0.025]], [9, [0.439,0.899]], [9, [0.173,0.985]]]
    # add 6s from images
    for el in img_list_6:
        query_list.append([6, full_process(el)])
    # add 9s from images
    for el in img_list_9:
        query_list.append([9, full_process(el)])
        
    i_c = 0
    for j in range(n_rep):
        for i, el in enumerate(query_list):
            print("{}: Features {}".format(el[0], el[1]))
            result = query(el[1])
            result_theo = query(el[1], mode_readout=1)
            if(result  == el[0]):
                n_correct += 1
            if(result_theo == el[0]):
                n_correct_t1 += 1
                
            i_c += 1

            print("[{}] Classified as {}; readout theo {}. Correct: {}; {}".format(i, result, \
                 result_theo, result  == el[0], result_theo == el[0]))
            print("So far {}; {} / {}".format(n_correct, n_correct_t1, i_c))

    print("Classified correctly {} / {}".format(n_correct, i_c))

def calc_theta(x_i):
    # ArcCot[z] is equal to ArcTan[1/z] for complex z, so also R
    theta = np.arctan(1/(x_i[0]/x_i[1]))
    return theta


def query(q, mode_readout=0):

    train6 = [0.987, 0.159]
    train9 = [0.354, 0.935]
   
    algo = QSVM(F, QVMConnection())
    algo.train(train6, train9)
    algo.verb_print_wavefunction = True
    
    if mode_readout == 0:
        # like hackathon, conditional wavefunction
        mes_rep = True
        mes_anc = True
    elif mode_readout == 1:
        # like theo paper (maybe?)
        # unclear what to do with ancilla
        mes_rep = False
        mes_anc =False
    else:
        raise ValueError('Unkown readout mode {}'.format(mode_readout))

    amplitudes = algo.run_quantum(q, mes_anc=mes_anc, mes_rep=mes_rep).amplitudes

    if mode_readout == 0:
        if(np.real(amplitudes[9]) > 0):
            res = 9
        else:
            res = 6
    
    elif mode_readout == 1:
        """ note: under construction """
        # classified as in theo paper, by projection on ancilla state
        p = 0 +0j
        p2 = 0 +0j
        norm = 0 +0j
        # gives all wf amplitudes with + if anc in 0, otherwise -
        psi_basis = np.array([1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1])/np.sqrt(16)
        #print(abs(np.dot(psi_basis, np.conj(amplitudes)))**2)
        psi2_basis = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])/np.sqrt(16)
        #print(abs(np.dot(psi2_basis, np.conj(psi_basis)))**2)
        #print(abs(amplitudes))

        p = abs(np.dot(psi_basis, amplitudes))**2
        """
        for i, el in enumerate(amplitudes):
            if i in [0, 2, 4, 6, 8, 10, 12, 14]:
                p += el/ np.sqrt(16)
                p2 += el/ np.sqrt(16)
                norm += el*np.conj(el)
            else:
                p -= el/ np.sqrt(16)
                p2 += el/ np.sqrt(16)
                norm += el*np.conj(el)
        
        p = abs(p)**2
        #p = abs(np.dot(psi_basis,amplitudes))**2
        p2 = abs(p2/np.sqrt(16))**2
        #p2 = abs(np.dot(psi2_basis,amplitudes))**2
        norm = abs(norm)**2
        """

        if p < 0.015: # empirical value !?  # for sqrt(2)
            res = 9
        else:
            res = 6
        print("P_a(|0>-|1>): {} -> {}".format(p, res))
        #print("P2_a(|0>+|1>): {} / P+P2: {:.2f} / P/(P+P2) {}".format(p2, p+p2, p/(p+p2)))

    return res



main()
        



        
        

        
        
        


        
