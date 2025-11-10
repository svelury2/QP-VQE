#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.optimize import minimize
import pybobyqa
import time

import cirq
import qsimcirq
import sympy as sp
import openfermion as of

import cupy as cp
from cuquantum import custatevec as cusv

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

#Bit Ordering in Cirq
#|01234567>
#For example, the state |00000001> corresponds to the site with index 7 being occupied, while the state |10000000> corresponds to the site with index 0 
#occupied

#Qubits on Lattice
m = 4
qubits = list(range(2**m));

#Interaction Links
links = [(i, (i+1) % (2**m)) for i in range(2**m)] #Periodic Boundaries

#XXZ Model Parameters
J = 4.0
Delta = 0.9

##Hamiltonian Construction

#Qubit Register for Hamiltonian
qreg = cirq.NamedQubit.range(2**m, prefix="q")

#Construct Pauli Strings
XX_links_strings = []
for (i, j) in links:
    XX_string = []
    for q in range(2**m):
        if q != i and q != j:
           XX_string.append(cirq.I(qreg[q]))
        else:
           XX_string.append(cirq.X(qreg[q]))
    # Convert list of characters to a string
    XX_links_strings.append(XX_string)


YY_links_strings = []
for (i, j) in links:
    YY_string = []
    for q in range(2**m):
        if q != i and q != j:
           YY_string.append(cirq.I(qreg[q]))
        else:
           YY_string.append(cirq.Y(qreg[q]))
    # Convert list of characters to a string
    YY_links_strings.append(YY_string)


ZZ_links_strings = []
for (i, j) in links:
    ZZ_string = []
    for q in range(2**m):
        if q != i and q != j:
           ZZ_string.append(cirq.I(qreg[q]))
        else:
           ZZ_string.append(cirq.Z(qreg[q]))
    #Convert list of characters to a string
    ZZ_links_strings.append(ZZ_string)

#Convert Pauli Strings to Many-Body Operators
XX_links_terms = [] 
for XX_link in XX_links_strings:
    XX_links_terms.append(cirq.PauliString(XX_link))

YY_links_terms = [] 
for YY_link in YY_links_strings:
    YY_links_terms.append(cirq.PauliString(YY_link))

ZZ_links_terms = [] 
for ZZ_link in ZZ_links_strings:
    ZZ_links_terms.append(cirq.PauliString(ZZ_link))

#Add Up All Terms
H_XXZ = 0
for XX_link in XX_links_terms:
    H_XXZ += 0.25*J*XX_link
for YY_link in YY_links_terms:
    H_XXZ += 0.25*J*YY_link
for ZZ_link in ZZ_links_terms:
    H_XXZ += 0.25*J*Delta*ZZ_link

#Define (Inverse) General Fourier Transform Gate (Change k to k + 1/2 for anti-periodic boundary conditions)
#def FT2kGateInverse(qubit_a, qubit_b, k, m):
    #return [
        #cirq.CZ(qubit_a, qubit_b),
        #of.circuits.FSWAP(qubit_a, qubit_b),
        #cirq.H(qubit_a),
        #cirq.H(qubit_b),
        #cirq.CNOT(qubit_a, qubit_b),
        #cirq.H(qubit_a),
        #cirq.H(qubit_b),
        #cirq.ry(np.pi/4).on(qubit_b),
        #cirq.CNOT(qubit_a, qubit_b),
        #cirq.ry(-np.pi/4).on(qubit_b),
        #cirq.H(qubit_a),
        #cirq.H(qubit_b),
        #cirq.CNOT(qubit_a, qubit_b),
        #cirq.H(qubit_a),
        #cirq.H(qubit_b),
        #of.circuits.FSWAP(qubit_a, qubit_b),
        #cirq.PhasedXZGate(x_exponent = 0, z_exponent = -2*k/(2**m), axis_phase_exponent = 0).on(qubit_a)
    #]

def FT2kGateInverse(qubit_a, qubit_b, k, m):
    return [
       cirq.rz(np.pi/2).on(qubit_a),
       cirq.H(qubit_a),
       cirq.H(qubit_b),
       cirq.CNOT(qubit_a, qubit_b),
       cirq.rz(np.pi/4).on(qubit_b),
       cirq.H(qubit_a),
       cirq.CNOT(qubit_a, qubit_b),
       cirq.rz(np.pi/2).on(qubit_a),
       cirq.rz(-np.pi/4).on(qubit_b),
       cirq.H(qubit_a),
       cirq.CNOT(qubit_a, qubit_b),
       cirq.H(qubit_a),
       cirq.H(qubit_b),
       cirq.rz(2*np.pi*k/(2**m)).on(qubit_a),
       cirq.rz(np.pi/2).on(qubit_b)
    ]

#Define General Fourier Transform Gate (Change k to k + 1/2 for anti-periodic boundary conditions)
#def FT2kGate(qubit_a, qubit_b, k, m):
    #return [
        #cirq.PhasedXZGate(x_exponent = 0, z_exponent = 2*k/(2**m), axis_phase_exponent = 0).on(qubit_a),
        #of.circuits.FSWAP(qubit_a, qubit_b),
        #cirq.H(qubit_a),
        #cirq.H(qubit_b),
        #cirq.CNOT(qubit_a, qubit_b),
        #cirq.H(qubit_a),
        #cirq.H(qubit_b),
        #cirq.ry(np.pi/4).on(qubit_b),
        #cirq.CNOT(qubit_a, qubit_b),
        #cirq.ry(-np.pi/4).on(qubit_b),
        #cirq.H(qubit_a),
        #cirq.H(qubit_b),
        #cirq.CNOT(qubit_a, qubit_b),
        #cirq.H(qubit_a),
        #cirq.H(qubit_b),
        #of.circuits.FSWAP(qubit_a, qubit_b),
        #cirq.CZ(qubit_a, qubit_b)
    #]

def FT2kGate(qubit_a, qubit_b, k, m):
    return [
       cirq.rz(-2*np.pi*k/(2**m)).on(qubit_a),
       cirq.rz(-np.pi/2).on(qubit_b),
       cirq.H(qubit_a),
       cirq.H(qubit_b),
       cirq.CNOT(qubit_a, qubit_b),
       cirq.H(qubit_a),
       cirq.rz(np.pi/4).on(qubit_b),
       cirq.rz(-np.pi/2).on(qubit_a),
       cirq.CNOT(qubit_a, qubit_b),
       cirq.H(qubit_a),
       cirq.rz(-np.pi/4).on(qubit_b),
       cirq.CNOT(qubit_a, qubit_b),
       cirq.H(qubit_a),
       cirq.H(qubit_b),
       cirq.rz(-np.pi/2).on(qubit_a)
    ]
        
#Define RXX, RYY, and RZZ gate functions
def RXX_gate(qubit_a, qubit_b, theta):
    return [
        cirq.H(qubit_a),
        cirq.H(qubit_b),
        cirq.CNOT(qubit_a, qubit_b),
        cirq.rz(-theta).on(qubit_b),
        cirq.CNOT(qubit_a, qubit_b),
        cirq.H(qubit_a),
        cirq.H(qubit_b),
    ]
def RYY_gate(qubit_a, qubit_b, theta):
    return [
        cirq.rx(-np.pi/2).on(qubit_a),
        cirq.rx(-np.pi/2).on(qubit_b),
        cirq.CNOT(qubit_a, qubit_b),
        cirq.rz(-theta).on(qubit_b),
        cirq.CNOT(qubit_a, qubit_b),
        cirq.rx(np.pi/2).on(qubit_a),
        cirq.rx(np.pi/2).on(qubit_b),
    ]
def RZZ_gate(qubit_a, qubit_b, theta):
    return [
        cirq.CNOT(qubit_a, qubit_b),
        cirq.rz(-theta).on(qubit_b),
        cirq.CNOT(qubit_a, qubit_b),
    ]

##Hamiltonian Variational Ansatz Construction & Initial State Preparation
def HVA_circuit(n_layers, theta_1_syms, theta_2_syms, theta_3_syms):
    global m
    global links
    global qubits

    import math
    
    #Initialize Circuit and Qubit Register (Pauli qubits in Momentum Space)
    qreg_HVA = cirq.NamedQubit.range(2**m, prefix="q")
    HVA = cirq.Circuit()
    N = len(qreg_HVA)

    #Construct Excited State in Momentum-Space 

    #Seven Fermion Momentum State (Periodic Boundaries, Superposition of q = 0 states)
    HVA.append(cirq.I(qreg_HVA[0])) #pi/8
    HVA.append(cirq.I(qreg_HVA[1])) #pi/4
    HVA.append(cirq.I(qreg_HVA[2])) #3pi/8
    HVA.append(cirq.I(qreg_HVA[3])) #pi/2
    HVA.append(cirq.I(qreg_HVA[4])) #5pi/8
    HVA.append(cirq.X(qreg_HVA[5])) #3pi/4
    HVA.append(cirq.X(qreg_HVA[6])) #7pi/8
    HVA.append(cirq.X(qreg_HVA[7])) #pi
    HVA.append(cirq.X(qreg_HVA[8])) #-7pi/8
    HVA.append(cirq.X(qreg_HVA[9])) #-3pi/4
    HVA.append(cirq.X(qreg_HVA[10])) #-5pi/8
    HVA.append(cirq.I(qreg_HVA[11])) #-pi/2
    HVA.append(cirq.X(qreg_HVA[12])) #-3pi/8
    HVA.append(cirq.I(qreg_HVA[13])) #-pi/4
    HVA.append(cirq.I(qreg_HVA[14])) #-pi/8
    HVA.append(cirq.I(qreg_HVA[15])) #0
    HVA.append(cirq.H(qreg_HVA[2]))
    HVA.append(cirq.CNOT(qreg_HVA[2], qreg_HVA[4]))
    HVA.append(cirq.CNOT(qreg_HVA[2], qreg_HVA[10]))
    HVA.append(cirq.CNOT(qreg_HVA[2], qreg_HVA[12]))
    
    
    #16 Qubit Inverse FFFT
    HVA.append(of.circuits.FSWAP(qreg_HVA[7], qreg_HVA[8]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[6], qreg_HVA[7]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[8], qreg_HVA[9]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[5], qreg_HVA[6]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[7], qreg_HVA[8]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[9], qreg_HVA[10]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[4], qreg_HVA[5]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[6], qreg_HVA[7]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[8], qreg_HVA[9]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[10], qreg_HVA[11]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[3], qreg_HVA[4]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[5], qreg_HVA[6]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[7], qreg_HVA[8]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[9], qreg_HVA[10]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[11], qreg_HVA[12]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[2], qreg_HVA[3]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[4], qreg_HVA[5]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[6], qreg_HVA[7]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[8], qreg_HVA[9]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[10], qreg_HVA[11]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[12], qreg_HVA[13]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[1], qreg_HVA[2]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[3], qreg_HVA[4]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[5], qreg_HVA[6]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[7], qreg_HVA[8]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[9], qreg_HVA[10]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[11], qreg_HVA[12]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[13], qreg_HVA[14]))
    HVA.append(FT2kGateInverse(qreg_HVA[0], qreg_HVA[1], 7, 4))
    HVA.append(FT2kGateInverse(qreg_HVA[2], qreg_HVA[3], 6, 4))
    HVA.append(FT2kGateInverse(qreg_HVA[4], qreg_HVA[5], 5, 4))
    HVA.append(FT2kGateInverse(qreg_HVA[6], qreg_HVA[7], 4, 4))
    HVA.append(FT2kGateInverse(qreg_HVA[8], qreg_HVA[9], 3, 4))
    HVA.append(FT2kGateInverse(qreg_HVA[10], qreg_HVA[11], 2, 4))
    HVA.append(FT2kGateInverse(qreg_HVA[12], qreg_HVA[13], 1, 4))
    HVA.append(FT2kGateInverse(qreg_HVA[14], qreg_HVA[15], 0, 4))
    HVA.append(of.circuits.FSWAP(qreg_HVA[13], qreg_HVA[14]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[11], qreg_HVA[12]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[9], qreg_HVA[10]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[7], qreg_HVA[8]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[5], qreg_HVA[6]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[3], qreg_HVA[4]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[1], qreg_HVA[2]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[12], qreg_HVA[13]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[10], qreg_HVA[11]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[8], qreg_HVA[9]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[6], qreg_HVA[7]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[4], qreg_HVA[5]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[2], qreg_HVA[3]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[11], qreg_HVA[12]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[9], qreg_HVA[10]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[7], qreg_HVA[8]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[5], qreg_HVA[6]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[3], qreg_HVA[4]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[10], qreg_HVA[11]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[8], qreg_HVA[9]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[6], qreg_HVA[7]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[4], qreg_HVA[5]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[9], qreg_HVA[10]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[7], qreg_HVA[8]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[5], qreg_HVA[6]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[8], qreg_HVA[9]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[6], qreg_HVA[7]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[7], qreg_HVA[8]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[11], qreg_HVA[12]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[3], qreg_HVA[4]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[12], qreg_HVA[13]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[10], qreg_HVA[11]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[4], qreg_HVA[5]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[2], qreg_HVA[3]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[13], qreg_HVA[14]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[11], qreg_HVA[12]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[9], qreg_HVA[10]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[5], qreg_HVA[6]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[3], qreg_HVA[4]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[1], qreg_HVA[2]))
    HVA.append(FT2kGateInverse(qreg_HVA[8], qreg_HVA[9], 3, 3))
    HVA.append(FT2kGateInverse(qreg_HVA[10], qreg_HVA[11], 2, 3))
    HVA.append(FT2kGateInverse(qreg_HVA[12], qreg_HVA[13], 1, 3))
    HVA.append(FT2kGateInverse(qreg_HVA[14], qreg_HVA[15], 0, 3))
    HVA.append(FT2kGateInverse(qreg_HVA[0], qreg_HVA[1], 3, 3))
    HVA.append(FT2kGateInverse(qreg_HVA[2], qreg_HVA[3], 2, 3))
    HVA.append(FT2kGateInverse(qreg_HVA[4], qreg_HVA[5], 1, 3))
    HVA.append(FT2kGateInverse(qreg_HVA[6], qreg_HVA[7], 0, 3))
    HVA.append(of.circuits.FSWAP(qreg_HVA[13], qreg_HVA[14]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[11], qreg_HVA[12]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[9], qreg_HVA[10]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[5], qreg_HVA[6]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[3], qreg_HVA[4]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[1], qreg_HVA[2]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[12], qreg_HVA[13]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[10], qreg_HVA[11]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[4], qreg_HVA[5]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[2], qreg_HVA[3]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[11], qreg_HVA[12]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[3], qreg_HVA[4]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[13], qreg_HVA[14]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[9], qreg_HVA[10]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[5], qreg_HVA[6]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[1], qreg_HVA[2]))
    HVA.append(FT2kGateInverse(qreg_HVA[0], qreg_HVA[1], 1, 2))
    HVA.append(FT2kGateInverse(qreg_HVA[2], qreg_HVA[3], 0, 2))
    HVA.append(FT2kGateInverse(qreg_HVA[4], qreg_HVA[5], 1, 2))
    HVA.append(FT2kGateInverse(qreg_HVA[6], qreg_HVA[7], 0, 2))
    HVA.append(FT2kGateInverse(qreg_HVA[8], qreg_HVA[9], 1, 2))
    HVA.append(FT2kGateInverse(qreg_HVA[10], qreg_HVA[11], 0, 2))
    HVA.append(FT2kGateInverse(qreg_HVA[12], qreg_HVA[13], 1, 2))
    HVA.append(FT2kGateInverse(qreg_HVA[14], qreg_HVA[15], 0, 2))
    HVA.append(of.circuits.FSWAP(qreg_HVA[13], qreg_HVA[14]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[9], qreg_HVA[10]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[5], qreg_HVA[6]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[1], qreg_HVA[2]))
    HVA.append(FT2kGateInverse(qreg_HVA[0], qreg_HVA[1], 0, 1))
    HVA.append(FT2kGateInverse(qreg_HVA[2], qreg_HVA[3], 0, 1))
    HVA.append(FT2kGateInverse(qreg_HVA[4], qreg_HVA[5], 0, 1))
    HVA.append(FT2kGateInverse(qreg_HVA[6], qreg_HVA[7], 0, 1))
    HVA.append(FT2kGateInverse(qreg_HVA[8], qreg_HVA[9], 0, 1))
    HVA.append(FT2kGateInverse(qreg_HVA[10], qreg_HVA[11], 0, 1))
    HVA.append(FT2kGateInverse(qreg_HVA[12], qreg_HVA[13], 0, 1))
    HVA.append(FT2kGateInverse(qreg_HVA[14], qreg_HVA[15], 0, 1))
    HVA.append(of.circuits.FSWAP(qreg_HVA[13], qreg_HVA[14]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[9], qreg_HVA[10]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[5], qreg_HVA[6]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[1], qreg_HVA[2]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[11], qreg_HVA[12]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[3], qreg_HVA[4]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[12], qreg_HVA[13]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[10], qreg_HVA[11]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[4], qreg_HVA[5]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[2], qreg_HVA[3]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[13], qreg_HVA[14]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[11], qreg_HVA[12]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[9], qreg_HVA[10]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[5], qreg_HVA[6]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[3], qreg_HVA[4]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[1], qreg_HVA[2]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[7], qreg_HVA[8]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[6], qreg_HVA[7]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[8], qreg_HVA[9]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[5], qreg_HVA[6]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[7], qreg_HVA[8]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[9], qreg_HVA[10]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[4], qreg_HVA[5]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[6], qreg_HVA[7]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[8], qreg_HVA[9]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[10], qreg_HVA[11]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[3], qreg_HVA[4]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[5], qreg_HVA[6]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[7], qreg_HVA[8]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[9], qreg_HVA[10]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[11], qreg_HVA[12]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[2], qreg_HVA[3]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[4], qreg_HVA[5]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[6], qreg_HVA[7]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[8], qreg_HVA[9]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[10], qreg_HVA[11]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[12], qreg_HVA[13]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[1], qreg_HVA[2]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[3], qreg_HVA[4]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[5], qreg_HVA[6]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[7], qreg_HVA[8]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[9], qreg_HVA[10]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[11], qreg_HVA[12]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[13], qreg_HVA[14]))

    #Declare Variational Parameters and Ansatz Construction (ZZ, XX, YY Blocks)
    for idx_layer in range(n_layers):
        theta_1 = theta_1_syms[idx_layer]
        theta_2 = theta_2_syms[idx_layer]
        theta_3 = theta_3_syms[idx_layer]
        for (iz, jz) in links:
            HVA.append(RZZ_gate(qreg_HVA[iz], qreg_HVA[jz], 2*theta_2))
        for (ix, jx) in links:
            HVA.append(RXX_gate(qreg_HVA[ix], qreg_HVA[jx], 2*theta_1))
        for (iy, jy) in links:
            HVA.append(RYY_gate(qreg_HVA[iy], qreg_HVA[jy], 2*theta_3))
            
    return HVA

##Construct Objective Function
def objective(params, hamiltonian, qreg, simulator): 
    #Resolve Parameters
    resolver = cirq.ParamResolver({all_symbols[i]: params[i]
                                   for i in range(len(all_symbols))})
    expect = simulator.simulate_expectation_values(base_ansatz, hamiltonian, resolver)
    return np.real(expect)
    
##Perform VQE

#Specify number of layers, iterations, and symbols (pre-build ansatz)
n_layers = 8
n_iter = 1600
theta_1_syms = [sp.Symbol(f"theta_1_{i}") for i in range(n_layers)]
theta_2_syms = [sp.Symbol(f"theta_2_{i}") for i in range(n_layers)]
theta_3_syms = [sp.Symbol(f"theta_3_{i}") for i in range(n_layers)]
all_symbols = theta_1_syms + theta_2_syms + theta_3_syms
base_ansatz = HVA_circuit(n_layers, theta_1_syms, theta_2_syms, theta_3_syms)

#VQE Functions for BOBYQA
def init_worker():
    #Each process must build its own GPU simulator 
    global simulator
    gpu_opts = qsimcirq.QSimOptions(use_gpu=True, gpu_mode=1)
    simulator = qsimcirq.QSimSimulator(qsim_options=gpu_opts)

def vqe_worker(_):
    #Random init & solve
    x_init = np.random.uniform(0, 2*np.pi, size=3*n_layers)
    res = pybobyqa.solve(objective,
                         x0=x_init,
                         args=(H_XXZ, qreg, simulator),
                         maxfun=100000)
    return res.f, res.nf, x_init, res.x

if __name__ == "__main__":
    # --- set up multiprocessing ---
    mp_ctx = mp.get_context("spawn")
    n_workers = max(1, mp_ctx.cpu_count() - 1)
    start_time = time.time()
    
    # Option A: using multiprocessing.Pool
    with mp_ctx.Pool(processes=n_workers, initializer=init_worker) as pool:
        results = pool.map(vqe_worker, range(n_iter))
    
    # --- unpack results ---
    optimal_values, cost_function_evals_collection, \
      initial_values_collection, optimal_parameters_collection = zip(*results)
      
    # --- convert to NumPy arrays ---
    optimal_values = np.array(optimal_values)                      # shape (n_iter,)
    cost_function_evals = np.array(cost_function_evals_collection) # shape (n_iter,)
    initial_vals = np.vstack(initial_values_collection)            # shape (n_iter, 2*n_layers)
    optimal_params = np.vstack(optimal_parameters_collection)      # shape (n_iter, 2*n_layers)
    
    end_time = time.time()
    
    print(f"n_layers: {n_layers}")
    print(f"Optimal Value: {min(optimal_values)}")
    best_idx = np.argmin(optimal_values)
    print(f"Optimal Parameters: {optimal_parameters_collection[best_idx]}")
    print(f"Cost Function Evals: {cost_function_evals_collection[best_idx]}")
    print(f"Elapsed Time: {end_time - start_time:.1f}s")
    print("*" * 40)
    
    ##Save Output Data
    np.savetxt('16_Qubit_XXZ_8_Layer_HVA_Initial_Parameters_BOBYQA_7_Filled_States_Periodic_BC_PH_State_8_Three_Parameters_Delta_0.9.txt', initial_values_collection)
    np.savetxt('16_Qubit_XXZ_8_Layer_HVA_Cost_Function_Evaluations_BOBYQA_7_Filled_States_Periodic_BC_PH_State_8_Three_Parameters_Delta_0.9.txt', cost_function_evals_collection)
    np.savetxt('16_Qubit_XXZ_8_Layer_HVA_Optimal_Values_BOBYQA_7_Filled_States_Periodic_BC_PH_State_8_Three_Parameters_Delta_0.9.txt', optimal_values)
    np.savetxt('16_Qubit_XXZ_8_Layer_HVA_Optimal_Parameters_BOBYQA_7_Filled_States_Periodic_BC_PH_State_8_Three_Parameters_Delta_0.9.txt', optimal_parameters_collection)

#VQE Functions for SciPy Optimizers

#Finite Difference Functions (For BFGS Only)
#def finite_diff_gradient(func, params, *args, epsilon=1e-6, **kwargs):
    #grad = np.zeros_like(params)
    #for i in range(len(params)):
        #params_eps_plus = np.copy(params)
        #params_eps_minus = np.copy(params)
        #params_eps_plus[i] += epsilon
        #params_eps_minus[i] -= epsilon
        #Convert the result to a scalar using .item()
        #f_plus = np.asarray(func(params_eps_plus, *args, **kwargs)).item()
        #f_minus = np.asarray(func(params_eps_minus, *args, **kwargs)).item()
        #grad[i] = (f_plus-f_minus)/(2*epsilon)
    #return grad

#def energy_grad_finite_diff(params, *args, **kwargs):
    #return finite_diff_gradient(objective, params, *args, epsilon=1e-6, **kwargs)
    
#def init_worker():
    #Each process must build its own GPU simulator 
    #global simulator
    #gpu_opts = qsimcirq.QSimOptions(use_gpu=True, gpu_mode=1)
    #simulator = qsimcirq.QSimSimulator(qsim_options=gpu_opts)

#def vqe_worker(_):

    # initial random parameters & solve
    #x_init = np.random.uniform(0, 2*np.pi, size=3*n_layers)
    #opt = {'maxiter': 10000}
    #res = minimize(objective, x0 = x_init, args=(H_XXZ, qreg, simulator), method = 'L-BFGS-B', jac=energy_grad_finite_diff, options=opt)

    #evaluate optimal value and optimal parameters
    #optimal_value = res.fun
    #cost_function_evaluations = res.nfev
    #optimal_parameters = res.x

    #return optimal_value, x_init, cost_function_evaluations, optimal_parameters
    
#if __name__ == "__main__":
    #mp_ctx = mp.get_context("spawn")
    #n_workers = max(1, mp_ctx.cpu_count() - 1)

    #start_time = time.time()

    #with mp_ctx.Pool(processes=n_workers, initializer=init_worker) as pool:
        #results = pool.map(vqe_worker, range(n_iter))

    #Unpack results
    #optimal_values, initial_values_collection, cost_function_evaluations_collection, optimal_parameters_collection = zip(*results)

    #Convert to arrays
    #optimal_values = np.array(optimal_values)
    #cost_function_evaluations_collection = np.array(cost_function_evaluations_collection)
    #initial_vals = np.vstack(initial_values_collection)
    #optimal_params = np.vstack(optimal_parameters_collection)

    #end_time = time.time()

    #Summary print
    #best_idx = np.argmin(optimal_values)
    #print(f"n_layers: {n_layers}")
    #print(f"Optimal Value: {optimal_values[best_idx]}")
    #print(f"Cost Function Evaluations: {cost_function_evaluations_collection[best_idx]}")
    #print(f"Optimal Parameters: {optimal_params[best_idx]}")
    #print(f"Elapsed time: {end_time - start_time:.1f}s")

    ##Save Output Data
    #np.savetxt('16_Qubit_XXZ_5_Layer_HVA_Initial_Parameters_L-BFGS-B_8_Filled_States_Antiperiodic_Boundary_Conditions_Three_Parameters_Delta_0.9.txt', initial_values_collection)
    #np.savetxt('16_Qubit_XXZ_5_Layer_HVA_Cost_Function_Evaluations_L-BFGS-B_8_Filled_States_Antiperiodic_Boundary_Conditions_Three_Parameters_Delta_0.9.txt', cost_function_evaluations_collection)
    #np.savetxt('16_Qubit_XXZ_5_Layer_HVA_Optimal_Values_L-BFGS-B_8_Filled_States_Antiperiodic_Boundary_Conditions_Three_Parameters_Delta_0.9.txt', optimal_values)
    #np.savetxt('16_Qubit_XXZ_5_Layer_HVA_Optimal_Parameters_L-BFGS-B_8_Filled_States_Antiperiodic_Boundary_Conditions_Three_Parameters_Delta_0.9.txt', optimal_parameters_collection)
 
 #ED Results Reference (Delta = 0.1):
#-21.17607219
#-20.73087244
#-20.73087244
#-19.62259632
#-19.61132073
#-19.52033917
#-19.52033917
#-19.41471952
#-19.41471952
#-19.20159526
#-19.20159526
#-19.20159526
#-19.20159526
#-19.10918402
#-19.10918402
#-19.10918402
#-19.10918402
#-18.09509873
#-18.09509873
#-18.06574551
#-18.0078685
#-18.0078685
#-17.98718208
#-17.98718208
#-17.97386693
#-17.97386693
#-17.97386693
#-17.97386693
#-17.89316564
#-17.89316564
#-17.89316564
#-17.89316564
#-17.8733048
#-17.81279038
#-17.81279038
#-17.81279038
#-17.81279038
#-17.74205868
#-17.74205868
#-17.74205868
#-17.74205868
#-17.60117039
#-17.60117039
#-17.56063577
#-17.56063577
#-17.49511915
#-17.49511915
#-17.49354881
#-17.49354881
#-17.49354881
#-17.49354881
#-17.28445286
#-17.28445286
#-16.91699487
#-16.91699487
#-16.87584619
#-16.87584619
#-16.87584619
#-16.87584619
#-16.87575423
#-16.87575423
#-16.85844197
#-16.85844197
#-16.83756548
#-16.83756548
#-16.83756548
#-16.83756548
#-16.82371574
#-16.82371574

 #ED Results Reference (Delta = 0.5):
 #-24.17105327
 #-23.49125499
 #-23.49125499
 #-22.73296598
 #-22.56772491
 #-22.15278005
 #-22.15278005
 #-21.99648994
 #-21.99648994
 #-21.99648994
 #-21.99648994
 #-21.53390982
 #-21.53390982
 #-21.53390982
 #-21.53390982
 #-21.5076736
 #-21.5076736
 #-20.90649092
 #-20.90649092
 #-20.60185142
 #-20.60185142
 #-20.36149142
 #-20.36149142
 #-20.28471179
 #-20.28471179
 #-20.27533616
 #-20.27533616
 #-20.26271695
 #-20.26271695
 #-20.26271695
 #-20.26271695
 #-20.24428142
 #-20.08849972
 #-20.08849972
 #-20.08849972
 #-20.08849972
 #-19.91057314
 #-19.91057314
 #-19.91057314
 #-19.91057314
 #-19.82379688
 #-19.70629296
 #-19.70629296
 #-19.70629296
 #-19.70629296
 #-19.6253931
 #-19.6253931
 #-19.59937504
 #-19.59937504
 #-19.59937504
 #-19.59937504
 #-19.43091646
 #-19.43091646
 #-19.17558772
 #-19.08400599
 #-19.08400599
 #-19.06017526
 #-19.06017526
 #-19.06017526
 #-19.06017526
 #-19.05327928
 #-19.05327928
 #-19.00315396
 #-19.00315396
 #-18.87460697
 #-18.87460697
 #-18.87004051
 #-18.87004051
 #-18.87004051
 #-18.87004051
 
 #ED Results (Delta = 0.9):
 #-27.63160419
 #-26.64503831
 #-26.64503831
 #-26.46411624
 #-25.90245383
 #-25.25284521
 #-25.25284521
 #-25.22638786
 #-25.22638786
 #-25.22638786
 #-25.22638786
 #-24.4086737
 #-24.4086737
 #-24.4086737
 #-24.4086737
 #-24.2842793 
 #-24.2842793
 #-23.8663527
 #-23.8663527
 #-23.59100504
 #-23.59100504
 #-23.54457482
 #-23.54457482
 #-23.51073895
 #-23.18925566
 #-23.18925566
 #-23.17182995
 #-23.17182995
 #-23.17182995
 #-23.17182995
 #-23.07750148
 #-23.07750148
 #-22.55914495
 #-22.55914495
 #-22.55914495
 #-22.55914495
 #-22.48592439
 #-22.48592439
 #-22.48592439
 #-22.48592439
 #-22.4703669
 #-22.4703669
 #-22.33416383
 #-22.33416383
 #-22.31144898
 #-22.31144898
 #-22.31144898
 #-22.31144898
 #-22.25471718
 #-22.25471718
 #-22.23157932
 #-22.23157932
 #-22.18608729
 #-22.02206133
 #-21.84955873
 #-21.84955873
 #-21.84955873
 #-21.84955873
 #-21.7413114
 #-21.7413114
 #-21.72524152
 #-21.72524152
 #-21.72524152
 #-21.72524152
 #-21.72511495
 #-21.72511495
 #-21.61476109
 #-21.61476109
 #-21.61476109
 #-21.61476109