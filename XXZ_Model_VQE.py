#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
import math
import pybobyqa
from scipy.optimize import minimize
import time

import cirq
import qsimcirq
import sympy as sp
import openfermion as of
import multiprocessing

#%%-----------------------------------------------------------
# Qubits on Lattice & XXZ Hamiltonian Setup
#--------------------------------------------------------------
m = 3
qubits = list(range(2**m))
links = [(i, (i+1) % (2**m)) for i in range(2**m)]  # periodic boundaries

# XXZ Model Parameters
J = 4.0
Delta = 0.5

# Build qubit register for Hamiltonian
qreg = cirq.NamedQubit.range(2**m, prefix="q")

# Build Pauli strings for each term
XX_links_strings = []
for (i, j) in links:
    XX_string = []
    for q in range(2**m):
        if q not in [i, j]:
            XX_string.append(cirq.I(qreg[q]))
        else:
            XX_string.append(cirq.X(qreg[q]))
    XX_links_strings.append(XX_string)

YY_links_strings = []
for (i, j) in links:
    YY_string = []
    for q in range(2**m):
        if q not in [i, j]:
            YY_string.append(cirq.I(qreg[q]))
        else:
            YY_string.append(cirq.Y(qreg[q]))
    YY_links_strings.append(YY_string)

ZZ_links_strings = []
for (i, j) in links:
    ZZ_string = []
    for q in range(2**m):
        if q not in [i, j]:
            ZZ_string.append(cirq.I(qreg[q]))
        else:
            ZZ_string.append(cirq.Z(qreg[q]))
    ZZ_links_strings.append(ZZ_string)

# Convert strings to PauliString operators
XX_links_terms = [cirq.PauliString(x) for x in XX_links_strings]
YY_links_terms = [cirq.PauliString(y) for y in YY_links_strings]
ZZ_links_terms = [cirq.PauliString(z) for z in ZZ_links_strings]

# Build Hamiltonian H_XXZ (using periodic boundary conditions)
H_XXZ = 0
for term in XX_links_terms:
    H_XXZ += 0.25 * J * term
for term in YY_links_terms:
    H_XXZ += 0.25 * J * term
for term in ZZ_links_terms:
    H_XXZ += 0.25 * J * Delta * term

#%%-----------------------------------------------------------
# Define Gates and Ansatz (HVA) Circuit
#--------------------------------------------------------------
#Define (Inverse) General Fourier Transform Gate (Change k to k + 1/2 for anti-periodic boundary conditions)
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

def HVA_circuit(n_layers):
    # We build the circuit starting from a predefined state preparation.
    qreg_HVA = [cirq.NamedQubit(f'q{i}') for i in qubits]
    circuit = cirq.Circuit()

    #Prepare a (hard-coded) initial state in momentum space:
    circuit.append(cirq.I(qreg_HVA[0]))
    circuit.append(cirq.I(qreg_HVA[1]))
    circuit.append(cirq.I(qreg_HVA[2]))
    circuit.append(cirq.X(qreg_HVA[3]))
    circuit.append(cirq.I(qreg_HVA[4]))
    circuit.append(cirq.I(qreg_HVA[5]))
    circuit.append(cirq.I(qreg_HVA[6]))
    circuit.append(cirq.I(qreg_HVA[7]))
    circuit.append(cirq.H(qreg_HVA[0]))
    circuit.append(cirq.CNOT(qreg_HVA[0], qreg_HVA[2]))
    circuit.append(cirq.X(qreg_HVA[0]))
    circuit.append(cirq.CNOT(qreg_HVA[0], qreg_HVA[4]))
    circuit.append(cirq.CNOT(qreg_HVA[0], qreg_HVA[6]))
    circuit.append(cirq.X(qreg_HVA[0]))
    circuit.append(cirq.Z(qreg_HVA[0]))

    #(Here follows a series of FSWAPs and FT2kGateInverse gates that implement the inverse FFFT)
    circuit.append(of.circuits.FSWAP(qreg_HVA[3], qreg_HVA[4]))
    circuit.append(of.circuits.FSWAP(qreg_HVA[2], qreg_HVA[3]))
    circuit.append(of.circuits.FSWAP(qreg_HVA[4], qreg_HVA[5]))
    circuit.append(of.circuits.FSWAP(qreg_HVA[1], qreg_HVA[2]))
    circuit.append(of.circuits.FSWAP(qreg_HVA[3], qreg_HVA[4]))
    circuit.append(of.circuits.FSWAP(qreg_HVA[5], qreg_HVA[6]))
    circuit.append(FT2kGateInverse(qreg_HVA[0], qreg_HVA[1], 3, 3))
    circuit.append(FT2kGateInverse(qreg_HVA[2], qreg_HVA[3], 2, 3))
    circuit.append(FT2kGateInverse(qreg_HVA[4], qreg_HVA[5], 1, 3))
    circuit.append(FT2kGateInverse(qreg_HVA[6], qreg_HVA[7], 0, 3))
    circuit.append(of.circuits.FSWAP(qreg_HVA[5], qreg_HVA[6]))
    circuit.append(of.circuits.FSWAP(qreg_HVA[3], qreg_HVA[4]))
    circuit.append(of.circuits.FSWAP(qreg_HVA[1], qreg_HVA[2]))
    circuit.append(of.circuits.FSWAP(qreg_HVA[4], qreg_HVA[5]))
    circuit.append(of.circuits.FSWAP(qreg_HVA[2], qreg_HVA[3]))
    circuit.append(of.circuits.FSWAP(qreg_HVA[3], qreg_HVA[4]))
    circuit.append(of.circuits.FSWAP(qreg_HVA[1], qreg_HVA[2]))
    circuit.append(of.circuits.FSWAP(qreg_HVA[5], qreg_HVA[6]))
    circuit.append(FT2kGateInverse(qreg_HVA[0], qreg_HVA[1], 1, 2))
    circuit.append(FT2kGateInverse(qreg_HVA[2], qreg_HVA[3], 0, 2))
    circuit.append(FT2kGateInverse(qreg_HVA[4], qreg_HVA[5], 1, 2))
    circuit.append(FT2kGateInverse(qreg_HVA[6], qreg_HVA[7], 0, 2))
    circuit.append(of.circuits.FSWAP(qreg_HVA[1], qreg_HVA[2]))
    circuit.append(of.circuits.FSWAP(qreg_HVA[5], qreg_HVA[6]))
    circuit.append(FT2kGateInverse(qreg_HVA[0], qreg_HVA[1], 0, 1))
    circuit.append(FT2kGateInverse(qreg_HVA[2], qreg_HVA[3], 0, 1))
    circuit.append(FT2kGateInverse(qreg_HVA[4], qreg_HVA[5], 0, 1))
    circuit.append(FT2kGateInverse(qreg_HVA[6], qreg_HVA[7], 0, 1))
    circuit.append(of.circuits.FSWAP(qreg_HVA[1], qreg_HVA[2]))
    circuit.append(of.circuits.FSWAP(qreg_HVA[5], qreg_HVA[6]))
    circuit.append(of.circuits.FSWAP(qreg_HVA[3], qreg_HVA[4]))
    circuit.append(of.circuits.FSWAP(qreg_HVA[2], qreg_HVA[3]))
    circuit.append(of.circuits.FSWAP(qreg_HVA[4], qreg_HVA[5]))
    circuit.append(of.circuits.FSWAP(qreg_HVA[1], qreg_HVA[2]))
    circuit.append(of.circuits.FSWAP(qreg_HVA[3], qreg_HVA[4]))
    circuit.append(of.circuits.FSWAP(qreg_HVA[5], qreg_HVA[6]))

    #Declare variational parameters (two per layer)
    theta_1 = sp.symarray('theta_1', n_layers)
    theta_2 = sp.symarray('theta_2', n_layers)

    #Ansatz: for each layer, add two-qubit rotations along each link.
    for idx_layer in range(n_layers):
        for (iz, jz) in links:
            circuit.append(RZZ_gate(qreg_HVA[iz], qreg_HVA[jz], 2*theta_2[idx_layer]))
        for (ix, jx) in links:
            circuit.append(RXX_gate(qreg_HVA[ix], qreg_HVA[jx], 2*theta_1[idx_layer]))
        for (iy, jy) in links:
            circuit.append(RYY_gate(qreg_HVA[iy], qreg_HVA[jy], 2*theta_1[idx_layer]))
          
    return circuit

#%%-----------------------------------------------------------
# Define Objective and Gradient Functions
#--------------------------------------------------------------
def objective(params, hamiltonian, qreg, simulator):
    # Build the ansatz circuit for the given number of layers.
    ansatz = HVA_circuit(n_layers)
    # Map parameters to their names.
    params_dict = {}
    for idx_layer in range(n_layers):
        params_dict[f'theta_1_{idx_layer}'] = params[idx_layer]
        params_dict[f'theta_2_{idx_layer}'] = params[n_layers + idx_layer]
    expect = simulator.simulate_expectation_values(ansatz, hamiltonian, params_dict)
    energy = np.real(expect)
    return energy


#def finite_diff_gradient(func, params, epsilon=1e-6):
    #grad = np.zeros_like(params)
    #for i in range(len(params)):
        #params_eps_plus = np.copy(params)
        #params_eps_minus = np.copy(params)
        #params_eps_plus[i] += epsilon
        #params_eps_minus[i] -= epsilon
        #f_plus = func(params_eps_plus)
        #f_minus = func(params_eps_minus)
        #grad[i] = (f_plus - f_minus) / (2 * epsilon)
    #return grad

#def energy_grad_finite_diff(params, *args):
    # Create a version of objective that only requires params.
    #wrapped_obj = lambda p: objective(p, *args)
    #return finite_diff_gradient(wrapped_obj, params)


#%%-----------------------------------------------------------
# Optimization Worker Functions for BFGS and COBYLA
#--------------------------------------------------------------
#def optimize_BFGS(args):
    """
    Each process runs one BFGS optimization from a random initial guess.
    The args tuple contains:
      (x_init, n_layers, opt_options, m)
    """
    #x_init, n_layers_local, opt_options, m_local = args
    # Reinitialize the simulator in this worker process.
    #options = {'t': 16, 'v': 3}
    #simulator = qsimcirq.QSimSimulator(options)
    # Rebuild the qreg for this worker.
    #qreg_local = cirq.NamedQubit.range(2**m_local, prefix="q")
    # Here we use the global H_XXZ (which must be defined at module scope).
    #res = minimize(
        #objective,
        #x0=x_init,
        #args=(H_XXZ, qreg_local, simulator),
        #method='BFGS',
        #jac=energy_grad_finite_diff,
        #options=opt_options
    #)
    #return res
    
#def optimize_COBYLA(args):
    """
    Each process runs one COBYLA optimization from a random initial guess.
    The args tuple contains:
      (x_init, n_layers, opt_options, m)
    """
    #x_init, n_layers_local, opt_options, m_local = args
    # Reinitialize the simulator in this worker process.
    #options = {'t': 16, 'v': 3}
    #simulator = qsimcirq.QSimSimulator(options)
    # Rebuild the qreg for this worker.
    #qreg_local = cirq.NamedQubit.range(2**m_local, prefix="q")
    # Here we use the global H_XXZ (which must be defined at module scope).
    #res = minimize(
        #objective,
        #x0 = x_init,
        #args = (H_XXZ, qreg_local, simulator),
        #method = 'COBYLA',
        #options = opt_options
    #)
    #return res
 
def optimize_BOBYQA(args):
    """
    Each process runs one BOBYQA optimization from a random initial guess.
    The args tuple contains:
        (x_init, n_layers, opt_options, m)
    """
    x_init, n_layers_local, opt_options, m_local = args
    # Reinitialize the simulator in this worker process.
    options = {'t': 16, 'v': 3}
    simulator = qsimcirq.QSimSimulator(options)
    # Rebuild the qreg for this worker.
    qreg_local = cirq.NamedQubit.range(2**m_local, prefix="q")
    # Here we use the global H_XXZ (which must be defined at module scope).
    res = pybobyqa.solve(
        objective,
        x0=x_init,
        args=(H_XXZ, qreg_local, simulator),
        maxfun=opt_options
    )
    return res
    
#%%-----------------------------------------------------------
# Main Parallel VQE Optimization
#--------------------------------------------------------------
if __name__ == '__main__':
    # Specify number of layers and number of optimization iterations (each with a random start)
    n_layers = 7
    n_iter = 800
    #opt_options = {'maxiter': 10000}
    opt_options = 10000

    # Prepare list of random initial parameters. (Dimension = 2*n_layers.)
    initial_params_list = [np.random.uniform(0, 2*np.pi, size=2*n_layers)
                             for _ in range(n_iter)]
    
    # Note: We no longer pass H_XXZ in the tuple.
    args_list = [(x_init, n_layers, opt_options, m) for x_init in initial_params_list]
    start_time = time.time()
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        #results = pool.map(optimize_BFGS, args_list)
        #results = pool.map(optimize_COBYLA, args_list)
        results = pool.map(optimize_BOBYQA, args_list)
    end_time = time.time()
    
    # (Then the rest of your code remains unchanged.)
    optimal_values = [res.f for res in results]
    cost_function_evals_collection = [res.nf for res in results]
    optimal_parameters_collection = [res.x for res in results]
    
    best_index = np.argmin(optimal_values)
    print(f"n_layers: {n_layers}")
    print(f"Optimal Value: {optimal_values[best_index]}")
    print(f"Optimal Parameters: {optimal_parameters_collection[best_index]}")
    print(f"Cost Function Evaluations: {cost_function_evals_collection[best_index]}")
    print(f"Elapsed Time: {end_time - start_time} seconds")
    print("*" * 40)
    
    # Save output data
    np.savetxt('8_Qubit_XXZ_7_Layer_HVA_Initial_Parameters_BOBYQA_GPU_3_Filled_States_Periodic_Boundary_Conditions_PH_State_4_Alternate_Delta_0.5.txt', np.array(initial_params_list))
    np.savetxt('8_Qubit_XXZ_7_Layer_HVA_Optimal_Values_BOBYQA_GPU_3_Filled_States_Periodic_Boundary_Conditions_PH_State_4_Alternate_Delta_0.5.txt', np.array(optimal_values))
    np.savetxt('8_Qubit_XXZ_7_Layer_HVA_Optimal_Parameters_BOBYQA_GPU_3_Filled_States_Periodic_Boundary_Conditions_PH_State_4_Alternate_Delta_0.5.txt', np.array(optimal_parameters_collection))
    np.savetxt('8_Qubit_XXZ_7_Layer_HVA_CF_Evaluations_BOBYQA_GPU_3_Filled_States_Periodic_Boundary_Conditions_PH_State_4_Alternate_Delta_0.5.txt', np.array(cost_function_evals_collection))
