# Estimate Bounds of Quantum Uncommon Information by Quantum Neural Network

This repository includes the simulation code and corresponding data for the paper "Estimate Bounds of Quantum Uncommon Information by Quantum Neural Network."

Detailed descriptions of each script are provided below.

1. `QUINE.py`
   - This code has two modules for estimate von Neumann entropy of the given state.
   - `quine` makes a random density matrix by given `num_qubits`, and estimate the entropy.
   - `quine_2` estimates the entropy of given state `psi`.
     
2. `Loose_bound.py`
   - Make a `A_qubit + B_qubit` density matrix.
   - For the upper bound, estimate `S(AB)`.
   - For the lower bound, estimate `|S(B)-S(A)|`.
   - The results are recorded as `data/loose_XX_N.csv`.
   
3. `Tight_ub.py`
   - Make a `A_qubit + B_qubit + R_qubit + A_qubit + B_qubit` density matrix with `0 ~ k-1` is a basis for the common subspace of A and B.
   - Estimate `S(AR)-S(A)`.
   - The results are recorded as `data/tight_ub_N.csv`.

4. `Tight_lb.py`
   - Make 3 EPR states and 1 GHZ state.
   - Estimate the tight lower bound.
   - The results are recorded as `data/tight_lb.csv`.

5. `Both_ub.py`
   - Estimate the tight and loose upper bound.
   - The results are recorded as `data/both_ub.csv`.

6. `make_fig.py`
   - For a given `.csv` file, make a figure
   - The figure is saved in the `fig` folder.
7. make_fig_both.py
   - For a given `both_ub.csv` file, make a figure
   - The figure is saved in the `fig` folder.
