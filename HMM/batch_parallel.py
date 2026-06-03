"""
Parallelise generation and fitting of HMM data.

Rik Henson
"""

import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from hmm_simulate_data import generate_data
from hmm_model_comparison import run_comparison

smooth_decliners = 0.0
SNR_levels = [1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.7, 3, 3.25, 3.5, 3.75, 4.25, 4.5, 4.75, 5]
#SNR_levels = [1, 2, 3, 4, 5]

def run_seed(seed):
    result = {}
    for snr in SNR_levels:
        path = f'simulation_data_{seed}_{snr}.npz'
        generate_data(SEED=seed, DFILE=path, SNR=snr, SMOOTH_DECLINER_FRAC=smooth_decliners)
        result[snr] = run_comparison(path)
        os.remove(path)
        print(f'Done noise SNR={snr}')
    return seed, result

num_seeds = 1000

best = {}
with ProcessPoolExecutor() as executor:
    futures = {executor.submit(run_seed, seed): seed for seed in range(1, num_seeds + 1)}
    for future in as_completed(futures):
        seed, result = future.result()
        best[seed] = result
        #print(f'Done seed {seed}')

print(best)

# calculate proportion of balues of K that are 2
proportion_K_2 = {}
proportion_K_2_or_more = {}
for snr in SNR_levels:
    print(f'At SNR of {snr}')
    proportion_K_2[snr] = sum(1 for seed_result in best.values() if seed_result.get(snr) == 2) / len(best)
    print(f'Proportion of K==2: {proportion_K_2[snr]:.2f}')
    proportion_K_2_or_more[snr] = sum(1 for seed_result in best.values() if seed_result.get(snr) >= 2) / len(best)
    print(f'Proportion of K>=2: {proportion_K_2_or_more[snr]:.2f}')

np.savetxt('power_analysis_results.csv',
           np.array([SNR_levels,
                     [proportion_K_2[snr] for snr in SNR_levels],
                     [proportion_K_2_or_more[snr] for snr in SNR_levels]]).T,
           delimiter=',', header='SNR,K==2,K>=2', comments='', fmt='%.4f')