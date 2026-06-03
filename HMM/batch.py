from random import seed
import numpy as np
from hmm_simulate_data import generate_data
from hmm_model_comparison import run_comparison

smooth_decliners = 0.0
SNR_levels = [2, 4]
num_seeds = 5

# For Figures (need to set PRINT_FLAG=1 in both scripts)
generate_data(SEED=1, SNR=8, SMOOTH_DECLINER_FRAC=0.0)
run_comparison()

best = {}
for s in range(1,num_seeds+1):
    result = {}
    for snr in SNR_levels:
        generate_data(SEED=s, SNR=snr, SMOOTH_DECLINER_FRAC=smooth_decliners)
        result[snr] = run_comparison()
        print(f'Done noise SNR={snr}')
    best[s] = result

print(best)

proportion_K_2 = {}
proportion_K_2_or_more = {}
for snr in SNR_levels:
    print(f'At SNR of {snr}')
    proportion_K_2[snr] = sum(1 for seed_result in best.values() if seed_result.get(snr) == 2) / len(best)
    print(f'Proportion of K==2: {proportion_K_2[snr]:.2f}')
    proportion_K_2_or_more[snr] = sum(1 for seed_result in best.values() if seed_result.get(snr) >= 2) / len(best)
    print(f'Proportion of K>=2: {proportion_K_2[snr]:.2f}')

np.savetxt('power_analysis_results.csv',
           np.array([SNR_levels,
                     [proportion_K_2[snr] for snr in SNR_levels],
                     [proportion_K_2_or_more[snr] for snr in SNR_levels]]).T,
           delimiter=',', header='SNR,K==2,K>=2', comments='', fmt='%.4f')
