#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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

import cupy as cp
from cuquantum import custatevec as cusv

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

#Bit Ordering in Cirq
#|01234567>
#For example, the state |00000001> corresponds to the site with index 7 being occupied, while the state |10000000> corresponds to the site with index 0 
#occupied

#Qubits on Lattice
m = 3
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
def HVA_circuit(n_layers):
    global m
    global links
    global qubits

    import math
    
    #Initialize Circuit and Qubit Register (Pauli qubits in Momentum Space)
    qreg_HVA = cirq.NamedQubit.range(2**m, prefix="q")
    HVA = cirq.Circuit()
    N = len(qreg_HVA)
    
    #Construct Excited State in Momentum-Space

    #Three Fermion Momentum State (Periodic Boundaries, Superposition of q = 0 states)
    HVA.append(cirq.I(qreg_HVA[0]))
    HVA.append(cirq.I(qreg_HVA[1]))
    HVA.append(cirq.I(qreg_HVA[2]))
    HVA.append(cirq.X(qreg_HVA[3]))
    HVA.append(cirq.X(qreg_HVA[4]))
    HVA.append(cirq.I(qreg_HVA[5]))
    HVA.append(cirq.X(qreg_HVA[6]))
    HVA.append(cirq.I(qreg_HVA[7]))
    HVA.append(cirq.H(qreg_HVA[0]))
    HVA.append(cirq.CNOT(qreg_HVA[0], qreg_HVA[2]))
    HVA.append(cirq.CNOT(qreg_HVA[0], qreg_HVA[4]))
    HVA.append(cirq.CNOT(qreg_HVA[0], qreg_HVA[6]))
    
    #8 Qubit Inverse FFFT
    HVA.append(of.circuits.FSWAP(qreg_HVA[3], qreg_HVA[4]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[2], qreg_HVA[3]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[4], qreg_HVA[5]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[1], qreg_HVA[2]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[3], qreg_HVA[4]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[5], qreg_HVA[6]))
    HVA.append(FT2kGateInverse(qreg_HVA[0], qreg_HVA[1], 3, 3))
    HVA.append(FT2kGateInverse(qreg_HVA[2], qreg_HVA[3], 2, 3))
    HVA.append(FT2kGateInverse(qreg_HVA[4], qreg_HVA[5], 1, 3))
    HVA.append(FT2kGateInverse(qreg_HVA[6], qreg_HVA[7], 0, 3))
    HVA.append(of.circuits.FSWAP(qreg_HVA[5], qreg_HVA[6]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[3], qreg_HVA[4]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[1], qreg_HVA[2]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[4], qreg_HVA[5]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[2], qreg_HVA[3]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[3], qreg_HVA[4]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[1], qreg_HVA[2]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[5], qreg_HVA[6]))
    HVA.append(FT2kGateInverse(qreg_HVA[0], qreg_HVA[1], 1, 2))
    HVA.append(FT2kGateInverse(qreg_HVA[2], qreg_HVA[3], 0, 2))
    HVA.append(FT2kGateInverse(qreg_HVA[4], qreg_HVA[5], 1, 2))
    HVA.append(FT2kGateInverse(qreg_HVA[6], qreg_HVA[7], 0, 2))
    HVA.append(of.circuits.FSWAP(qreg_HVA[1], qreg_HVA[2]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[5], qreg_HVA[6]))
    HVA.append(FT2kGateInverse(qreg_HVA[0], qreg_HVA[1], 0, 1))
    HVA.append(FT2kGateInverse(qreg_HVA[2], qreg_HVA[3], 0, 1))
    HVA.append(FT2kGateInverse(qreg_HVA[4], qreg_HVA[5], 0, 1))
    HVA.append(FT2kGateInverse(qreg_HVA[6], qreg_HVA[7], 0, 1))
    HVA.append(of.circuits.FSWAP(qreg_HVA[1], qreg_HVA[2]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[5], qreg_HVA[6]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[3], qreg_HVA[4]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[2], qreg_HVA[3]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[4], qreg_HVA[5]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[1], qreg_HVA[2]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[3], qreg_HVA[4]))
    HVA.append(of.circuits.FSWAP(qreg_HVA[5], qreg_HVA[6]))   
  
    #Declare Variational Parameters
    theta_1 = sp.symarray('theta_1', n_layers)
    theta_2 = sp.symarray('theta_2', n_layers)
    theta_3 = sp.symarray('theta_3', n_layers)

    #Ansatz Construction (ZZ, XX, YY Blocks)
    for idx_layer in range(n_layers):
        for (iz, jz) in links:
            HVA.append(RZZ_gate(qreg_HVA[iz], qreg_HVA[jz], 2*theta_2[idx_layer]))
        for (ix, jx) in links:
            HVA.append(RXX_gate(qreg_HVA[ix], qreg_HVA[jx], 2*theta_1[idx_layer]))
        for (iy, jy) in links:
            HVA.append(RYY_gate(qreg_HVA[iy], qreg_HVA[jy], 2*theta_3[idx_layer]))
                
    return HVA

##Construct Objective Function
def objective(params, hamiltonian, qreg, simulator):
    ansatz = HVA_circuit(n_layers)
    
    #Map params to parameter names dynamically
    params_dict = {}
    n_params = n_layers
    for idx_layer in range(n_layers):
        params_dict[f'theta_1_{idx_layer}'] = params[idx_layer]
        params_dict[f'theta_2_{idx_layer}'] = params[n_params + idx_layer]
        params_dict[f'theta_3_{idx_layer}'] = params[2*n_params + idx_layer]

    expect = simulator.simulate_expectation_values(ansatz, hamiltonian, params_dict)
    energy = np.real(expect)
    return energy

##Perform VQE

#Specify number of layers and iterations
n_layers = 8
n_iter = 800

#VQE Functions for BOBYQA
def vqe_worker(_):
    # each process must build its own GPU simulator & qubit register
    gpu_opts = qsimcirq.QSimOptions(use_gpu=True, gpu_mode=1)
    simulator = qsimcirq.QSimSimulator(qsim_options=gpu_opts)
    qreg = cirq.NamedQubit.range(2**m, prefix="q")
    
    # random init & solve
    x_init = np.random.uniform(0, 2*np.pi, size=3*n_layers)
    res = pybobyqa.solve(objective,
                         x0=x_init,
                         args=(H_XXZ, qreg, simulator),
                         maxfun=10000)
    return res.f, res.nf, x_init, res.x

if __name__ == "__main__":
    # --- set up multiprocessing ---
    mp_ctx = mp.get_context("spawn")
    n_workers = max(1, mp_ctx.cpu_count() - 1)
    
    start_time = time.time()
    
    # Option A: using multiprocessing.Pool
    with mp_ctx.Pool(processes=n_workers) as pool:
        results = pool.map(vqe_worker, range(n_iter))
    
    # Option B: using concurrent.futures
    # with ProcessPoolExecutor(mp_context=mp_ctx, max_workers=n_workers) as exe:
    #     futures = [exe.submit(vqe_worker, i) for i in range(n_iter)]
    #     results = [f.result() for f in futures]
    
    # --- unpack results ---
    optimal_values, cost_function_evals_collection, \
      initial_values_collection, optimal_parameters_collection = zip(*results)
      
    # --- convert to NumPy arrays ---
    optimal_values = np.array(optimal_values)                   # shape (n_iter,)
    initial_vals = np.vstack(initial_values_collection)        # shape (n_iter, 2*n_layers)
    optimal_params = np.vstack(optimal_parameters_collection)  # shape (n_iter, 2*n_layers)
    
    end_time = time.time()
    
    print(f"n_layers: {n_layers}")
    print(f"Optimal Value: {min(optimal_values)}")
    best_idx = np.argmin(optimal_values)
    print(f"Optimal Parameters: {optimal_parameters_collection[best_idx]}")
    print(f"Cost Function Evals: {cost_function_evals_collection[best_idx]}")
    print(f"Elapsed Time: {end_time - start_time:.1f}s")
    print("*" * 40)
    
    ##Save Output Data
    np.savetxt('8_Qubit_XXZ_8_Layer_HVA_Initial_Parameters_BOBYQA_GPU_3_Filled_States_Periodic_BC_PH_State_4_Three_Parameters_Delta_0.9.txt', initial_values_collection)
    np.savetxt('8_Qubit_XXZ_8_Layer_HVA_Cost_Function_Evaluations_BOBYQA_GPU_3_Filled_States_Periodic_BC_PH_State_4_Three_Parameters_Delta_0.9.txt', cost_function_evals_collection)
    np.savetxt('8_Qubit_XXZ_8_Layer_HVA_Optimal_Values_BOBYQA_GPU_3_Filled_States_Periodic_BC_PH_State_4_Three_Parameters_Delta_0.9.txt', optimal_values)
    np.savetxt('8_Qubit_XXZ_8_Layer_HVA_Optimal_Parameters_BOBYQA_GPU_3_Filled_States_Periodic_BC_PH_State_4_Three_Parameters_Delta_0.9.txt', optimal_parameters_collection)

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

#def vqe_worker(_):
    # each process builds its own GPU simulator & qubit register
    #gpu_opts = qsimcirq.QSimOptions(use_gpu=True, gpu_mode=1)
    #simulator = qsimcirq.QSimSimulator(qsim_options=gpu_opts)
    #qreg = cirq.NamedQubit.range(2**m, prefix="q")

    # initial random parameters & solve
    #x_init = np.random.uniform(0, 2*np.pi, size=2*n_layers)
    #opt = {'maxiter': 10000}
    #res = minimize(objective, x0 = x_init, args=(H_XXZ, qreg, simulator), method = 'BFGS', jac = energy_grad_finite_diff, options=opt)

    #evaluate optimal value and optimal parameters
    #optimal_value = res.fun
    #cost_function_evaluations = res.nfev
    #optimal_parameters = res.x

    #return optimal_value, x_init, cost_function_evaluations, optimal_parameters
    
#if __name__ == "__main__":
    #mp_ctx = mp.get_context("spawn")
    #n_workers = max(1, mp_ctx.cpu_count() - 1)

    #start_time = time.time()

    #with mp_ctx.Pool(processes=n_workers) as pool:
        #results = pool.map(vqe_worker, range(n_iter))

    # Unpack results
    #optimal_values, initial_values_collection, cost_function_evaluations_collection, optimal_parameters_collection = zip(*results)

    # Convert to arrays
    #optimal_values = np.array(optimal_values)
    #cost_function_evaluations_collection = np.array(cost_function_evaluations_collection)
    #initial_vals = np.vstack(initial_values_collection)
    #optimal_params = np.vstack(optimal_parameters_collection)

    #end_time = time.time()

    # Summary print
    #best_idx = np.argmin(optimal_values)
    #print(f"n_layers: {n_layers}")
    #print(f"Optimal Value: {optimal_values[best_idx]}")
    #print(f"Cost Function Evaluations: {cost_function_evaluations_collection[best_idx]}")
    #print(f"Optimal Parameters: {optimal_params[best_idx]}")
    #print(f"Elapsed time: {end_time - start_time:.1f}s")

    ##Save Output Data
    #np.savetxt('8_Qubit_XXZ_1_Layer_HVA_Initial_Parameters_BFGS_GPU_3_Filled_States_Periodic_Boundary_Conditions_PH_State_2.txt', initial_values_collection)
    #np.savetxt('8_Qubit_XXZ_1_Layer_HVA_Cost_Function_Evaluations_BFGS_GPU_3_Filled_States_Periodic_Boundary_Conditions_PH_State_2.txt', cost_function_evaluations_collection)
    #np.savetxt('8_Qubit_XXZ_1_Layer_HVA_Optimal_Values_BFGS_GPU_3_Filled_States_Periodic_Boundary_Conditions_PH_State_2.txt', optimal_values)
    #np.savetxt('8_Qubit_XXZ_1_Layer_HVA_Optimal_Parameters_BFGS_GPU_3_Filled_States_Periodic_Boundary_Conditions_PH_State_2.txt', optimal_parameters_collection)