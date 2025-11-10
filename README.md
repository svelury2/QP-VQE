# QP-VQE

This repository contains code used to generate the data and the plots in the manuscript "Quasiparticle Variational Quantum Eigensolver" (arXiv:25XX.XXXX). More details on the various packages used and other computational details are explained in Appendix A of the manuscript. Below is a brief description of the functionality of each script:

-"XXZ_Model_VQE": This script runs VQE using the BOBYQA optimizer for the 8-qubit XXZ model. It is designed to run noiseless statevector simulations on the CPU.
-"XXZ_Model_VQE_GPU": This script runs VQE using the BOBYQA optimizer for the 16-qubit XXZ model. It is designed to run noiseless statevector simulations on the GPU (using Nvidia's cuQuantum package).

Any questions please contact Saavanth Velury at velurys@ufl.edu.
