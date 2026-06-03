def generate_data(SEED = 1, DFILE = 'simulation_data.npz', SNR = 1, SMOOTH_DECLINER_FRAC = 0.0):
    """
    Written by Chetan Gohil. Adapted slightly by Rik Henson.

    Simulation of ageing trajectories with burst-decline dynamics.

    Inspired by Fjell et al. (2026) "Punctuated memory change: The temporal dynamics
    and brain basis of memory stability in aging."

    Demonstrates how smooth population-level memory decline can arise from
    heterogeneous individual trajectories characterised by periods of stability
    punctuated by brief episodes of accelerated loss ("bursts" / critical transitions).

    Parameters match the paper's Online Methods "Individual vs. group trajectory
    simulation" section:
    - Virtual participants observed annually, ages 30-90
    - Hypothetical memory test, range 0-20
    - Each participant starts near mean score ~16
    - 1-3 bursts per person, onset governed by age-dependent hazard
    - Burst duration 2-4 years, steeper slopes at older ages
    - Latent process → logistic sigmoid → bounded 0-20 scores
    - Small Gaussian measurement noise
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.ndimage import uniform_filter1d

    # -- Simulation parameters -----------------------------------------------

    PRINT_FLAG = 0
    N_PARTICIPANTS = 20
    AGE_MIN, AGE_MAX = 70, 77
    AGES = np.arange(AGE_MIN, AGE_MAX + 1, 1/6, dtype=float)  # observations every 2 months
    N_AGES = len(AGES)
    SCORE_FLOOR, SCORE_CEIL = 0.0, 200.0 # IQ
    SCORE_PLOT_FLOOR, SCORE_PLOT_CEIL = 40.0, 140.0

    # Participant type mixture
    # Fraction of "smooth decliners" (0 bursts, steady gradual decline)   
    SMOOTH_SLOPE_MEAN = -0.05          # latent decline / year (produces ~group-mean curve)
    SMOOTH_SLOPE_SD = 0.01

    # Latent-space intercept and slope
    # logistic(0) ≈ 0.50 → 0.5*SCORE_CEIL on test scale
    LATENT_INTERCEPT_MEAN = 0
    LATENT_INTERCEPT_SD = 0.1          # between-person variability
    LATENT_SLOPE_MEAN = 0       
    LATENT_SLOPE_SD = 0

    # Burst configuration (for burst-type participants)
    BURST_N_MIN, BURST_N_MAX = 1, 1
    BURST_DUR_MIN, BURST_DUR_MAX = 1/6, 1  # years (make bursts at least 2 timepoints for hmm assumptions?)
    BURST_MAG_BASE = 0.70               # latent drop / year at reference age 
    BURST_MAG_AGE_COEF = 0.0          # extra drop / year for each year above 
    BURST_EARLIEST_AGE = AGE_MIN            # no bursts before this age
    BURST_MIN_GAP = AGE_MAX - AGE_MIN                   # minimum years between burst onsets

    # Age-dependent hazard for burst timing
    HAZARD_INTERCEPT = 0.5            # annual hazard at BURST_EARLIEST_AGE
    HAZARD_SLOPE = 0 #1/(AGE_MAX - AGE_MIN)              # increase per year of age to ensure always one

    # -- Helper functions ----------------------------------------------------
    def logistic(z):
        """Sigmoid mapping latent z → (0, 1)."""
        return 1.0 / (1.0 + np.exp(-z))

    def latent_to_score(z):
        """Map latent value to bounded memory score."""
        return SCORE_CEIL * logistic(z)

    # SNR
    mean_dur = 0.5 * (BURST_DUR_MIN + BURST_DUR_MAX)
    score_drop = latent_to_score(0) - latent_to_score(-BURST_MAG_BASE * mean_dur)                                                                                                                       
    NOISE_SD = score_drop / SNR  # SD of measurement noise, set by signal-to-noise ratio
  
    def sample_burst_onsets(n_bursts, rng):
        """Sample burst onset ages from an age-dependent hazard, respecting gaps."""
        all_ages = AGES[AGES >= BURST_EARLIEST_AGE]
        hazard = HAZARD_INTERCEPT + HAZARD_SLOPE * (all_ages - AGE_MIN)
        hazard = np.clip(hazard, 0, None)

        onsets = []
        for _ in range(n_bursts):
            # Zero out ages too close to existing onsets
            mask = np.ones(len(all_ages), dtype=bool)
            for prev in onsets:
                mask &= np.abs(all_ages - prev) >= BURST_MIN_GAP
            if not mask.any():
                break
            weights = hazard[mask]
            candidate_ages = all_ages[mask]
            weights /= weights.sum()
            onset = rng.choice(candidate_ages, p=weights)
            onsets.append(float(onset))
            #r = rng.random(len(candidate_ages))
            #onset = candidate_ages[r<weights]
            #onsets.append(float(onset[0]))
        return sorted(onsets)


    def simulate_one(rng):
        """Simulate a single participant's trajectory. Returns (true_score, observed, n_bursts, onsets)."""
        intercept = rng.normal(LATENT_INTERCEPT_MEAN, LATENT_INTERCEPT_SD)

        # Decide participant type: smooth decliner vs burst-decline
        is_smooth = rng.random() < SMOOTH_DECLINER_FRAC

        if is_smooth:
            # Smooth, gradual decline — no bursts
            slope = rng.normal(SMOOTH_SLOPE_MEAN, SMOOTH_SLOPE_SD)
            n_bursts = 0
            onsets = []
            latent = intercept + slope * (AGES - AGE_MIN)
        else:
            # Burst-decline trajectory
            slope = rng.normal(LATENT_SLOPE_MEAN, LATENT_SLOPE_SD)
            n_bursts = rng.integers(BURST_N_MIN, BURST_N_MAX + 1)
            onsets = sample_burst_onsets(n_bursts, rng)
            durations = [rng.uniform(BURST_DUR_MIN, BURST_DUR_MAX) for _ in onsets]
            magnitudes = [BURST_MAG_BASE + BURST_MAG_AGE_COEF * (o - AGE_MIN) for o in onsets]

            latent = intercept + slope * (AGES - AGE_MIN)

            for onset, dur, mag in zip(onsets, durations, magnitudes):
                in_burst = (AGES >= onset) & (AGES < onset + dur)
                post_burst = AGES >= onset + dur
                total_drop = mag * dur
                if in_burst.any():
                    progress = (AGES[in_burst] - onset) / dur
                    latent[in_burst] -= total_drop * progress
                latent[post_burst] -= total_drop

        true_score = latent_to_score(latent)
        observed = true_score + rng.normal(0, NOISE_SD, N_AGES)
        #observed = np.clip(observed, SCORE_FLOOR, SCORE_CEIL)

        return true_score, observed, n_bursts, onsets


    # -- Run the simulation --------------------------------------------------
    print(f"Simulating {N_PARTICIPANTS} participants, ages {AGE_MIN}-{AGE_MAX}, seed {SEED}, snr {SNR} (Noise_SD = {NOISE_SD:.2f})...")
    rng = np.random.default_rng(SEED)

    true_scores = np.empty((N_PARTICIPANTS, N_AGES))
    observed_scores = np.empty((N_PARTICIPANTS, N_AGES))
    n_bursts_arr = np.empty(N_PARTICIPANTS, dtype=int)
    all_onsets = []

    for i in range(N_PARTICIPANTS):
        ts, obs, nb, ons = simulate_one(rng)
        true_scores[i] = ts
        observed_scores[i] = obs
        n_bursts_arr[i] = nb
        all_onsets.append(ons)

    # Save simulation data for downstream modelling
    np.savez(DFILE,
            observed=observed_scores, true=true_scores,
            ages=AGES, n_bursts=n_bursts_arr)
    print(f"Saved {DFILE}")

    # Group-level summaries
    group_mean = observed_scores.mean(axis=0)
    group_median = np.median(observed_scores, axis=0)
    smooth_mean = uniform_filter1d(group_mean, size=5)
    pct = {q: np.percentile(observed_scores, q, axis=0) for q in [5, 10, 25, 75, 90, 95]}

    if PRINT_FLAG:
        #print(f"  Mean score at age 50: {group_mean[AGES == 50][0]:.1f}")
        print(f"  Mean score at age 70: {group_mean[AGES == 70][0]:.1f}")
        #print(f"  Mean score at age 90: {group_mean[AGES == 90][0]:.1f}")

        print(f"  Type counts — smooth: {(n_bursts_arr==0).sum()}, "
            f"1 burst: {(n_bursts_arr==1).sum()}, "
            f"2 bursts: {(n_bursts_arr==2).sum()}, "
            f"3 bursts: {(n_bursts_arr==3).sum()}")

    # -- Figure 1: Main 3-panel figure (cf. paper Figure 8A) -----------------
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

        type_colours = {0: '#9C27B0', 1: '#2196F3', 2: '#FF9800', 3: '#4CAF50'}
        type_labels = {0: 'Smooth decline', 1: '1 burst', 2: '2 bursts', 3: '3 bursts'}

        # - Panel A: highlighted individual trajectories + group mean -
        ax = axes[0]
        n_highlight = 20
        highlighted = dict.fromkeys(range(BURST_N_MAX+1), None) #{}
        for nb in range(BURST_N_MAX+1):
            idx = np.where(n_bursts_arr == nb)[0]
            if len(idx) > 0:
                sel = rng.choice(idx, min(n_highlight, len(idx)), replace=False)
                highlighted[nb] = sel
                for k, j in enumerate(sel):
                    ax.plot(AGES, true_scores[j], color=type_colours[nb], alpha=0.85,
                        linewidth=1.4, label=type_labels[nb] if k == 0 else None)

        ax.plot(AGES, smooth_mean, 'k-', linewidth=3, label='Group mean')
        ax.set_xlabel('Age (years)')
        ax.set_ylabel('Memory score')
        ax.set_title('A   Individual burst-decline trajectories')
        ax.legend(loc='lower left', fontsize=9, framealpha=0.9)
        ax.set_ylim(SCORE_PLOT_FLOOR, SCORE_PLOT_CEIL)
        ax.set_xlim(AGE_MIN, AGE_MAX)

        # - Panel B: population-level curve with prediction bands -
        ax = axes[1]
        ax.fill_between(AGES, pct[10], pct[90], alpha=0.12, color='steelblue', label='10–90th %ile')
        ax.fill_between(AGES, pct[25], pct[75], alpha=0.28, color='steelblue', label='25–75th %ile')
        ax.plot(AGES, smooth_mean, 'k-', linewidth=3, label='Mean')
        ax.plot(AGES, group_median, '--', color='#c62828', linewidth=2, label='Median')
        ax.set_xlabel('Age (years)')
        ax.set_ylabel('Memory score')
        ax.set_title('B   Population-level memory decline')
        ax.legend(loc='lower left', fontsize=9, framealpha=0.9)
        ax.set_ylim(SCORE_PLOT_FLOOR, SCORE_PLOT_CEIL)
        ax.set_xlim(AGE_MIN, AGE_MAX)

        # - Panel C: spaghetti of subset of observed trajectories -
        ax = axes[2]
        n_highlight = 20
        spaghetti_idx = rng.choice(N_PARTICIPANTS, n_highlight, replace=False)
        for j in spaghetti_idx:
            ax.plot(AGES, observed_scores[j], alpha=0.25, color='steelblue', linewidth=0.7)
        ax.plot(AGES, smooth_mean, 'k-', linewidth=3, label='Group mean')
        ax.set_xlabel('Age (years)')
        ax.set_ylabel('Memory score')
        ax.set_title(f'C   {n_highlight} observed trajectories')
        ax.legend(loc='lower left', fontsize=9, framealpha=0.9)
        ax.set_ylim(SCORE_PLOT_FLOOR, SCORE_PLOT_CEIL)
        ax.set_xlim(AGE_MIN, AGE_MAX)

        plt.tight_layout()
        plt.savefig('figures/fig1_burst_decline_simulation.png', dpi=200, bbox_inches='tight')
        plt.close()
        print("Saved figures/fig1_burst_decline_simulation.png")


        # -- Figure 2: Detailed view of individual exemplars ---------------------
        n_highlight = 3
        fig, axes = plt.subplots(BURST_N_MAX+1, n_highlight, figsize=(15, 15), sharex=True, sharey=True)

        for row in range(BURST_N_MAX+1):
            sel = highlighted[row]
            if isinstance(sel, (list, np.ndarray)):
                for col in range(n_highlight):
                    ax = axes[row, col]
                    j = sel[col]
                    # Observed points
                    ax.scatter(AGES, observed_scores[j], s=12, color='grey', alpha=0.6, zorder=2)
                    # True (noise-free) trajectory
                    ax.plot(AGES, true_scores[j], color=type_colours[nb], linewidth=2, zorder=3)
                    # Mark burst onsets
                    for onset in all_onsets[j]:
                        ax.axvline(onset, color='red', linestyle=':', alpha=0.5, linewidth=1)
                    if col == 0:
                        ax.set_ylabel(f'{row} bursts\nMemory score')
                    if row == 3:
                        ax.set_xlabel('Age (years)')
                    ax.set_ylim(SCORE_PLOT_FLOOR, SCORE_PLOT_CEIL)

        fig.suptitle('Individual trajectories: true curve (colour) and observed scores (grey)\n'
                    'Red dotted lines mark burst onsets', fontsize=13, y=1.01)
        plt.tight_layout()
        #plt.savefig('figures/fig2_individual_exemplars.png', dpi=200, bbox_inches='tight')
        #plt.close()
        #print("Saved figures/fig2_individual_exemplars.png")


        # -- Figure 3: Burst-onset age distribution ------------------------------
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

        all_onset_ages = [o for ons in all_onsets for o in ons]

        ax = axes[0]
        ax.hist(all_onset_ages, bins=np.arange(AGE_MIN, AGE_MAX, 1), color='steelblue',
                edgecolor='white', alpha=0.8)
        ax.set_xlabel('Age at burst onset')
        ax.set_ylabel('Count')
        ax.set_title('A   Distribution of burst onsets')

        ax = axes[1]
        type_counts = [int((n_bursts_arr == b).sum()) for b in [0, 1, 2, 3]]
        bars = ax.bar([0, 1, 2, 3], type_counts,
                    color=['#9C27B0', '#2196F3', '#FF9800', '#4CAF50'], edgecolor='white', width=0.6)
        ax.set_xlabel('Number of bursts')
        ax.set_ylabel('Count')
        ax.set_title('B   Trajectory types')
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xticklabels(['Smooth', '1', '2', '3'])

        plt.tight_layout()
        #plt.savefig('figures/fig3_burst_distributions.png', dpi=200, bbox_inches='tight')
        #plt.close()
        #print("Saved figures/fig3_burst_distributions.png")

        print("\nDone.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--dfile", type=str, default='simulation_data.npz', help="Output path for simulated data")
    parser.add_argument("--snr", type=float, default=0.1, help="Mean Burst change relative to SD of noise")
    parser.add_argument("--smooth_decliners", type=float, default=0.0, help="Fraction of smooth decliners")
    args = parser.parse_args()
    generate_data(SEED=args.seed, DFILE=args.dfile, SNR=args.snr, SMOOTH_DECLINER_FRAC=args.smooth_decliners)
