"""
Written by Chetan Gohil. Adapted slightly by Rik Henson.

Fit linear regression HMMs with K states to simulated data.

Compare models via BIC and evaluate classification accuracy.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from hmm_fit import LinearRegressionHMM

# -- Settings ------------------------------------------------------------
K_VALUES = [1, 2, 3]
SEED = 42
PRINT_FLAG = 1

def max_consecutive_in_state(z, state):
    """Return the longest run of consecutive time points in a given state."""
    best = 0
    count = 0
    for s in z:
        if s == state:
            count += 1
            if count > best:
                best = count
        else:
            count = 0
    return best


def classify_accuracy(state_seqs, n_bursts_true, K, burst_state=None, min_duration=0):
    """Evaluate smooth-vs-burst classification from decoded states.

    If burst_state and min_duration are provided, an individual is classified
    as a burst decliner only if they spend >= min_duration consecutive time
    points in burst_state.  Otherwise, any state transition counts.
    """
    N = len(state_seqs)
    true_is_smooth = n_bursts_true == 0

    if K == 1:
        return {'accuracy': np.nan, 'sensitivity': np.nan, 'specificity': np.nan}

    if burst_state is not None and min_duration > 0:
        pred_has_burst = np.array([
            max_consecutive_in_state(z, burst_state) >= min_duration
            for z in state_seqs])
    else:
        pred_has_burst = np.array([len(np.unique(z)) > 1 for z in state_seqs])

    pred_is_smooth = ~pred_has_burst

    correct = (true_is_smooth == pred_is_smooth).sum()
    tp = (pred_has_burst & ~true_is_smooth).sum()
    fp = (pred_has_burst & true_is_smooth).sum()
    tn = (pred_is_smooth & true_is_smooth).sum()
    fn = (pred_is_smooth & ~true_is_smooth).sum()

    return {
        'accuracy': correct / N,
        'sensitivity': tp / max(tp + fn, 1),
        'specificity': tn / max(tn + fp, 1),
        'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
    }


def run_comparison(data_path='simulation_data.npz'):
    # -- Load data -------------------------------------------------------
    print(f'Loading {data_path}...')
    data = np.load(data_path)
    observed = data['observed']
    true_scores = data['true']
    ages = data['ages']
    n_bursts = data['n_bursts']

    rng = np.random.default_rng(SEED)
    N = len(observed)
    #print(f'  {N} participants, ages {ages[0]:.0f}-{ages[-1]:.0f}')
    #print(f'  Smooth: {(n_bursts==0).sum()}, Burst: {(n_bursts>0).sum()}')

    # Z-score across the full population (global mean & SD)
    global_mean = observed.mean()
    global_sd = observed.std()
    observed_z = (observed - global_mean) / global_sd
    #print(f'  Z-scored: mean={global_mean:.2f}, sd={global_sd:.2f}')

    # Build per-individual lists (all share the same ages here,
    # but the model supports irregular schedules)
    Y = [observed_z[i] for i in range(N)]
    A_ages = [ages.copy() for _ in range(N)]

    # -- Fit models ------------------------------------------------------
    results = {}

    for K in K_VALUES:
        if PRINT_FLAG:
            print(f'\n{"="*60}')
            print(f'Fitting K={K} states')
            print(f'{"="*60}')

        model = LinearRegressionHMM(n_states=K)
        ll_hist, state_seqs, alphas, deltas = model.fit(Y, A_ages, max_iter=80)

        #print(f'\nComputing BIC (GPB(1) marginal likelihood)...')
        bic_val, marg_ll = model.bic(Y, A_ages)

        acc = classify_accuracy(state_seqs, n_bursts, K)

        # Duration-filtered classification for K > 1
        #
        # A naive rule ("any visit to a non-baseline state ⇒ burst") is too
        # sensitive: noisy observations can flip a single time point into the
        # burst state for a truly smooth decliner, producing false positives.
        # We therefore require an individual to dwell in the burst state for
        # at least `min_dur_k` consecutive years before classifying them as a
        # burst decliner.
        #
        # The threshold is derived from the model itself rather than tuned by
        # hand.  We:
        #   1. identify the burst state as the one with the steepest (most
        #      negative) slope, beta;
        #   2. evaluate the transition matrix at the mean age and read off the
        #      self-transition probability p = A[burst, burst];
        #   3. compute the expected dwell time of a geometric distribution,
        #      E[duration] = 1 / (1 - p);
        #   4. floor it to an integer number of years, with a minimum of 2 so
        #      that a single noisy time point can never trigger a positive.
        acc_filt = acc  # default (K=1 or no burst state)
        if K > 1:
            burst_st = int(np.argmin(model.beta))
            A_mid_k = model.transition_matrix(model.age_mean)
            exp_dur = 1.0 / (1.0 - A_mid_k[burst_st, burst_st])
            min_dur_k = max(int(np.floor(exp_dur)), 2)
            acc_filt = classify_accuracy(state_seqs, n_bursts, K, burst_state=burst_st, min_duration=min_dur_k)

        results[K] = {
            'model': model,
            'bic': bic_val,
            'marginal_ll': marg_ll,
            'viterbi_ll': ll_hist[-1],
            'state_seqs': state_seqs,
            'alphas': alphas,
            'deltas': deltas,
            'll_history': ll_hist,
            'classification': acc,
            'classification_filtered': acc_filt,
        }

        if PRINT_FLAG:
            print(f'  BIC = {bic_val:.1f}')
            print(f'  Marginal LL = {marg_ll:.1f}')
            print(f'  Parameters: {model.n_parameters()}')
            print(f'  beta = {model.beta}')
            print(f'  sigma = {np.sqrt(model.sigma2):.3f}')
            print(f'  sd_delta = {np.sqrt(model.Sigma_re[1,1]):.4f}')
            A_mid = model.transition_matrix(model.age_mean)
            print(f'  Transition matrix (at mean age):\n{A_mid}')
            if K > 1:
                print(f'  Classification (filtered): acc={acc_filt["accuracy"]:.3f}  '
                    f'sens={acc_filt["sensitivity"]:.3f}  spec={acc_filt["specificity"]:.3f}')

    # -- Summary ---------------------------------------------------------
    #print(f'\n{"="*60}')
    #print('Model comparison summary')
    #print(f'{"="*60}')

    # Log Bayes factors relative to K=1 (baseline)
    ref_ll = results[1]['marginal_ll']
    print(f'{"K":>3s}  {"BIC":>12s}  {"Marg LL":>12s}  {"log BF vs K=1":>14s}  {"p":>4s}')
    for K in K_VALUES:
        r = results[K]
        log_bf = r['marginal_ll'] - ref_ll
        print(f'{K:3d}  {r["bic"]:12.1f}  {r["marginal_ll"]:12.1f}  {log_bf:14.1f}  {r["model"].n_parameters():4d}')
    print(f'{"="*60}')

    best_K_bic = min(K_VALUES, key=lambda k: results[k]['bic'])
    best_K_mll = max(K_VALUES, key=lambda k: results[k]['marginal_ll'])
    #print(f'\nBest model by BIC: K = {best_K_bic}')
    #print(f'Best model by marginal LL: K = {best_K_mll}')

    # For figure colouring, highlight BIC-best
    best_K = best_K_bic

    # -- Duration-filtered classification summary (best model) -----------
    if best_K > 1:
        best_model = results[best_K]['model']
        state_seqs_best = results[best_K]['state_seqs']

        burst_state = int(np.argmin(best_model.beta))
        A_mid = best_model.transition_matrix(best_model.age_mean)
        expected_dur = 1.0 / (1.0 - A_mid[burst_state, burst_state])
        min_dur = max(int(np.floor(expected_dur)), 2)

        if PRINT_FLAG:
            print(f'\n{"="*60}')
            print(f'Duration-filtered classification (K={best_K})')
            print(f'{"="*60}')
            print(f'  Burst state: {burst_state} (beta={best_model.beta[burst_state]:.4f})')
            print(f'  Expected burst duration at mean age: {expected_dur:.1f} yrs')
            print(f'  Minimum duration threshold: {min_dur} yrs')

            filt_acc = results[best_K]['classification_filtered']
            print(f'  Accuracy:    {filt_acc["accuracy"]:.3f}')
            print(f'  Sensitivity: {filt_acc["sensitivity"]:.3f}')
            print(f'  Specificity: {filt_acc["specificity"]:.3f}')

        # Compute across a range of thresholds for the figure
        thresholds = list(range(1, 8))
        thresh_results = []
        for thr in thresholds:
            tr = classify_accuracy(state_seqs_best, n_bursts, best_K,
                                   burst_state=burst_state, min_duration=thr)
            thresh_results.append(tr)

        results[best_K]['burst_state'] = burst_state
        results[best_K]['min_duration'] = min_dur

    if PRINT_FLAG:
        # -- Figure 1: Model comparison -------------------------------------
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

        bics = [results[K]['bic'] for K in K_VALUES]
        marg_lls = [results[K]['marginal_ll'] for K in K_VALUES]
        log_bfs = [results[K]['marginal_ll'] - ref_ll for K in K_VALUES]
        colours_bic = ['#4CAF50' if K == best_K_bic else '#90CAF9' for K in K_VALUES]
        colours_mll = ['#4CAF50' if K == best_K_mll else '#90CAF9' for K in K_VALUES]

        ax = axes[0]
        ax.bar(K_VALUES, bics, color=colours_bic, edgecolor='white', width=0.6)
        ax.set_xlabel('Number of states (K)')
        ax.set_ylabel('BIC (lower is better)')
        ax.set_title('A   BIC')
        ax.set_xticks(K_VALUES)

        ax = axes[1]
        ax.bar(K_VALUES, marg_lls, color=colours_mll, edgecolor='white', width=0.6)
        ax.set_xlabel('Number of states (K)')
        ax.set_ylabel('Marginal log-likelihood')
        ax.set_title('B   Model evidence (GPB(1))')
        ax.set_xticks(K_VALUES)

        ax = axes[2]
        bar_cols = ['grey' if K == 1 else '#FF9800' for K in K_VALUES]
        ax.bar(K_VALUES, log_bfs, color=bar_cols, edgecolor='white', width=0.6)
        ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
        ax.set_xlabel('Number of states (K)')
        ax.set_ylabel('log Bayes factor vs K=1')
        ax.set_title('C   Bayes factors')
        ax.set_xticks(K_VALUES)

        plt.tight_layout()
        plt.savefig('figures/fig4_model_comparison.png', dpi=200, bbox_inches='tight')
        plt.close()
        print('Saved figures/fig4_model_comparison.png')

        # -- Figure 2: Decoded trajectories from best model ------------------
        best = results[best_K]
        model = best['model']
        state_seqs = best['state_seqs']
        alphas_best = best['alphas']
        deltas_best = best['deltas']
        K = best_K

        # Distinct, high-contrast state colours
        state_cmap = ['#4CAF50', '#FFC107', '#E53935']  # green, amber, red
        if K > 3:
            state_cmap = plt.cm.tab10.colors[:K]
        T = len(ages)

        fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharex=True, sharey=True)

        smooth_idx = np.where(n_bursts == 0)[0]
        burst_idx = np.where(n_bursts > 0)[0]
        fig_rng = np.random.default_rng(99)  # fresh seed for reproducible selection

        # for col, i in enumerate(fig_rng.choice(smooth_idx, 4, replace=False)):
        #     ax = axes[0, col]
        #     z = state_seqs[i]
        #     for t in range(T - 1):
        #         ax.axvspan(ages[t], ages[t + 1], alpha=0.3, color=state_cmap[z[t]], lw=0)
        #     ax.scatter(ages, observed_z[i], s=10, color='k', alpha=0.5, zorder=2)
        #     pred = model._predicted(ages, z, model.beta, alphas_best[i], deltas_best[i])
        #     ax.plot(ages, pred, 'k-', lw=1.5, zorder=3)
        #     n_unique = len(np.unique(z))
        #     ax.set_title(f'true: smooth | states used: {n_unique}', fontsize=9)
        #     if col == 0:
        #         ax.set_ylabel('Smooth decliner\nz-score')

        for col, i in enumerate(fig_rng.choice(burst_idx, 4, replace=False)):
            ax = axes[1, col]
            z = state_seqs[i]
            for t in range(T - 1):
                ax.axvspan(ages[t], ages[t + 1], alpha=0.3, color=state_cmap[z[t]], lw=0)
            ax.scatter(ages, observed_z[i], s=10, color='k', alpha=0.5, zorder=2)
            pred = model._predicted(ages, z, model.beta, alphas_best[i], deltas_best[i])
            ax.plot(ages, pred, 'k-', lw=1.5, zorder=3)
            ax.set_title(f'true: {n_bursts[i]} burst(s) | states used: {len(np.unique(z))}',
                        fontsize=9)
            if col == 0:
                ax.set_ylabel('Burst decliner\nz-score')
            ax.set_xlabel('Age (years)')

        from matplotlib.patches import Patch
        legend_items = [Patch(facecolor=state_cmap[k], alpha=0.5,
                            label=f'State {k} (slope={model.beta[k]:.4f}/yr)')
                        for k in range(K)]
        legend_items.append(plt.Line2D([], [], color='k', lw=1.5, label='HMM fit'))
        fig.legend(handles=legend_items, loc='upper center', ncol=K + 1, fontsize=9,
                bbox_to_anchor=(0.5, 1.02))
        fig.suptitle(f'Best model: K={K} states — decoded trajectories (z-scored)',
                    fontsize=13, y=1.06)
        plt.tight_layout()
        plt.savefig('figures/fig5_decoded_trajectories.png', dpi=200, bbox_inches='tight')
        plt.close()
        print('Saved figures/fig5_decoded_trajectories.png')

        # -- Figure 3: Age-dependent transition probabilities ----------------
        if best_K > 1:
            age_grid = np.linspace(float(ages[0]), float(ages[-1]), 200)
            fig, axes = plt.subplots(1, K, figsize=(5 * K, 4), sharey=True)
            if K == 1:
                axes = [axes]

            state_names = [f'State {k}' for k in range(K)]
            line_colours = state_cmap[:K]

            for j in range(K):
                ax = axes[j]
                probs = np.array([model.transition_matrix(a)[j] for a in age_grid])
                for k in range(K):
                    ax.plot(age_grid, probs[:, k], color=line_colours[k], lw=2,
                            label=f'→ {state_names[k]}')
                ax.set_xlabel('Age (years)')
                if j == 0:
                    ax.set_ylabel('Transition probability')
                ax.set_title(f'From {state_names[j]}')
                ax.set_ylim(-0.02, 1.02)
                ax.legend(fontsize=8)

            fig.suptitle('Age-dependent transition probabilities', fontsize=13)
            plt.tight_layout()
            plt.savefig('figures/fig6_transition_probs.png', dpi=200, bbox_inches='tight')
            plt.close()
            print('Saved figures/fig6_transition_probs.png')

        # -- Figure 4: Decoding performance summary ---------------------------
        if best_K > 1 and 'burst_state' in results[best_K]:
            burst_st = results[best_K]['burst_state']
            min_dur_val = results[best_K]['min_duration']
            filt_c = results[best_K]['classification_filtered']

            fig, axes = plt.subplots(1, 3, figsize=(18, 5))

            # Panel A: accuracy, sensitivity, specificity
            ax = axes[0]
            metrics = ['Accuracy', 'Sensitivity', 'Specificity']
            filt_vals = [filt_c['accuracy'], filt_c['sensitivity'], filt_c['specificity']]
            x_m = np.arange(len(metrics))
            ax.bar(x_m, filt_vals, width=0.5, color='#4CAF50', edgecolor='white')
            ax.set_xticks(x_m)
            ax.set_xticklabels(metrics)
            ax.set_ylim(0, 1.05)
            ax.set_ylabel('Score')
            ax.set_title('A   Classification performance')
            for xi, fv in zip(x_m, filt_vals):
                ax.text(xi, fv + 0.02, f'{fv:.3f}', ha='center', fontsize=9)

            # Panel B: performance vs duration threshold
            ax = axes[1]
            accs_t = [t['accuracy'] for t in thresh_results]
            sens_t = [t['sensitivity'] for t in thresh_results]
            spec_t = [t['specificity'] for t in thresh_results]
            ax.plot(thresholds, accs_t, 'ko-', lw=2, label='Accuracy')
            ax.plot(thresholds, sens_t, 's--', color='#2196F3', lw=1.5, label='Sensitivity')
            ax.plot(thresholds, spec_t, '^--', color='#E53935', lw=1.5, label='Specificity')
            ax.axvline(min_dur_val, color='grey', ls=':', lw=1.5,
                    label=f'Model-derived threshold ({min_dur_val})')
            ax.set_xlabel('Minimum consecutive time points in burst state')
            ax.set_ylabel('Score')
            ax.set_title('B   Effect of duration threshold')
            ax.set_ylim(0, 1.05)
            ax.set_xticks(thresholds)
            ax.legend(fontsize=8)

            # Panel C: confusion matrix (filtered)
            ax = axes[2]
            cm = np.array([[filt_c['tn'], filt_c['fp']],
                            [filt_c['fn'], filt_c['tp']]])
            im = ax.imshow(cm, cmap='Blues', aspect='auto')
            for ci in range(2):
                for cj in range(2):
                    ax.text(cj, ci, f'{cm[ci, cj]}', ha='center', va='center',
                            fontsize=16, fontweight='bold',
                            color='white' if cm[ci, cj] > cm.max() * 0.6 else 'black')
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['Pred: Smooth', 'Pred: Burst'])
            ax.set_yticks([0, 1])
            ax.set_yticklabels(['True: Smooth', 'True: Burst'])
            ax.set_title('C   Confusion matrix (filtered)')

            fig.suptitle(f'Decoding performance — K={best_K}, '
                        f'burst state {burst_st} '
                        f'(expected duration {1/(1-A_mid[burst_st, burst_st]):.1f} yrs)',
                        fontsize=13)
            plt.tight_layout()
            plt.savefig('figures/fig7_decoding_performance.png', dpi=200, bbox_inches='tight')
            plt.close()
            print('Saved figures/fig7_decoding_performance.png')

        print('\nDone.')

    return best_K

if __name__ == '__main__':
    result = run_comparison()