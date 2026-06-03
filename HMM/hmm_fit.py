"""
Linear Regression HMM with random intercept + slope and age-dependent
transitions.

Model
-----
For individual i with observation y_{it} at age a_t and hidden state z_t:

    y_{it} = alpha_i + delta_i * (a_t - a_0)        # individual line
             + sum_{s<=t} beta_{z_s} * (a_s - a_{s-1})   # cumulative state slope
             + eps_{it},                            eps ~ N(0, sigma2)

    [alpha_i, delta_i] ~ N(mu_re, Sigma_re)         # random effects (intercept + slope)
    z_t | z_{t-1}, a_t ~ Categorical(A(a_t)[z_{t-1}, :])

    A_{jk}(age) = softmax_k( W[j, k, 0] + W[j, k, 1] * age_normalised )
    age_normalised = (age - age_mean) / age_sd

So each state contributes its own slope `beta_k` only while the chain is in
state k; the cumulative drop along the realised path is the sum of those
contributions.  An individual baseline (alpha_i) and an individual residual
slope (delta_i) are layered on top as random effects.

Algorithms
----------
* Training: Viterbi EM.  E-step decodes the most-likely state path per
  individual; M-step refits beta, sigma2, the transition weights W (via
  multinomial logistic regression), the initial distribution pi, and the
  random-effects hyperparameters (mu_re, Sigma_re).
* Marginal likelihood (for BIC): GPB(1) forward pass that integrates the
  random effects (alpha_i, delta_i) analytically using Kalman updates.
* Post-hoc decoding (optional): forward-backward over a pointwise emission
  model returning posterior marginals gamma[t, k].

Public API
----------
    model = LinearRegressionHMM(n_states=K)
    ll_hist, state_seqs, alphas, deltas = model.fit(Y, A_ages)
    bic_value, marginal_ll = model.bic(Y, A_ages)
    A_age = model.transition_matrix(age)

`Y` and `A_ages` are lists of 1-D arrays — one per individual — so the
model handles irregular observation schedules.
"""

import numpy as np
from scipy.optimize import minimize
from time import time


class LinearRegressionHMM:
    """Hierarchical linear-regression HMM with age-dependent transitions.

    Parameters
    ----------
    n_states : int
        Number of hidden slope-regime states K.

    Attributes (set by `fit`)
    -------------------------
    beta : (K,) ndarray
        State-specific slopes, sorted in descending order
        (state 0 = shallowest decline, state K-1 = steepest).
    sigma2 : float
        Observation-noise variance.
    pi : (K,) ndarray
        Initial-state distribution.
    W : (K, K, 2) ndarray
        Transition logits.  W[j, k, 0] is the intercept and W[j, k, 1]
        the age-effect coefficient for the j -> k transition.
    age_mean, age_sd : float
        Used to normalise age before applying the transition model.
    mu_re : (2,) ndarray
        Mean of the random-effects prior [alpha, delta].
    Sigma_re : (2, 2) ndarray
        Covariance of the random-effects prior.
    """

    def __init__(self, n_states):
        self.K = n_states
        # Group-level parameters (set by _initialise / fit)
        self.beta = None           # (K,) state-specific slopes
        self.sigma2 = None         # observation-noise variance
        self.pi = None             # (K,) initial-state distribution
        self.W = None              # (K, K, 2) age-dependent transition logits
        self.age_mean = 0.0        # used to normalise age before logits
        self.age_sd = 1.0
        # Random-effects prior over [alpha_i, delta_i]
        self.mu_re = None          # (2,)
        self.Sigma_re = None       # (2, 2)

    # -- Transition matrix at a given age --------------------------------

    def transition_matrix(self, age):
        """Return A(age) — the (K, K) transition matrix at the given age.

        Row j of the result is the conditional distribution P(z_t | z_{t-1}=j),
        obtained by softmax over the linear-in-age logits.
        """
        return np.exp(self.log_transition_matrix(age))

    def log_transition_matrix(self, age):
        """Numerically stable log A(age) — see `transition_matrix`."""
        age_n = (age - self.age_mean) / self.age_sd
        logits = self.W[:, :, 0] + self.W[:, :, 1] * age_n  # (K, K)
        # Row-wise log-softmax
        mx = logits.max(axis=1, keepdims=True)
        log_norm = mx + np.log(np.exp(logits - mx).sum(axis=1, keepdims=True))
        return logits - log_norm

    # -- Initialisation --------------------------------------------------

    def _initialise(self, Y, A_ages):
        """Set initial parameter values from the data.

        Strategy:
          * Compute every per-interval slope (dy/dt) across all individuals
            and use them to seed `beta` via 1-D K-means.
          * Use the empirical mean / SD of ages to set the age normaliser
            for the transition model.
          * Start with diagonal-dominant transitions and no age effect.
          * Set the random-effects prior from the empirical distribution of
            first observations and per-interval slopes.
        """
        K = self.K

        # -- Collect all interval slopes and all ages -------------------
        all_slopes = []
        all_ages = []
        for y, ages in zip(Y, A_ages):
            da = np.diff(ages)
            dy = np.diff(y)
            valid = da > 0
            all_slopes.extend((dy[valid] / da[valid]).tolist())
            all_ages.extend(ages.tolist())
        all_slopes = np.array(all_slopes)

        # Age normaliser used inside the transition model
        self.age_mean = float(np.mean(all_ages))
        self.age_sd = max(float(np.std(all_ages)), 1.0)

        # -- Seed state slopes via 1-D K-means on interval slopes --------
        if K == 1:
            self.beta = np.array([np.mean(all_slopes)])
        else:
            # Initial centres at evenly-spaced percentiles, then 50 K-means iters
            pcts = np.linspace(5, 95, K)
            centres = np.percentile(all_slopes, pcts)
            for _ in range(50):
                dists = np.abs(all_slopes[:, None] - centres[None, :])
                labels = np.argmin(dists, axis=1)
                for k in range(K):
                    mask = labels == k
                    if mask.any():
                        centres[k] = np.mean(all_slopes[mask])
            centres.sort()
            self.beta = centres[::-1]  # state 0 = shallowest decline

        self.sigma2 = np.var(all_slopes) * 0.5
        self.pi = np.ones(K) / K

        # -- Transition weights: diagonal-dominant, no age effect --------
        # W[j, j, 0] = 3.0  ⇒  softmax row for j gives ~95% self-transition.
        # W[:, :, 1] = 0    ⇒  no age dependence at initialisation.
        self.W = np.zeros((K, K, 2))
        for j in range(K):
            self.W[j, j, 0] = 3.0

        # -- Random-effects prior over [alpha_i, delta_i] ----------------
        first_obs = np.array([y[0] for y in Y])
        mu_alpha = float(np.mean(first_obs))
        var_alpha = float(np.var(first_obs)) + 1.0
        var_delta = float(np.var(all_slopes)) * 0.5
        self.mu_re = np.array([mu_alpha, 0.0])
        self.Sigma_re = np.diag([var_alpha, var_delta])

    # -- Cumulative slope helper -----------------------------------------

    @staticmethod
    def _cumulative_slopes(ages, states, beta):
        """Walk along the realised state path summing per-segment drops.

        Returns cum[t] = sum_{s<=t} beta[z_s] * (a_s - a_{s-1}), with cum[0]=0.
        """
        T = len(ages)
        cum = np.zeros(T)
        for t in range(1, T):
            cum[t] = cum[t - 1] + beta[states[t]] * (ages[t] - ages[t - 1])
        return cum

    @classmethod
    def _predicted(cls, ages, states, beta, alpha_i, delta_i):
        """Reconstruct the noise-free trajectory implied by a state path.

        y_hat_t = alpha_i + delta_i * (a_t - a_0) + cumulative_state_drops_t
        """
        cum = cls._cumulative_slopes(ages, states, beta)
        return alpha_i + delta_i * (ages - ages[0]) + cum

    # -- Viterbi (E-step) ------------------------------------------------

    def _viterbi(self, y, ages, alpha_i, delta_i, log_A_all=None):
        """Viterbi decoding for one individual, vectorised over states.

        Returns the most-likely state path and its joint log-probability under
        the cumulative-slope emission model.

        Implementation note
        -------------------
        Because the emission depends on the *cumulative* state slope along the
        path (not just the current state), each Viterbi message must remember
        the cumulative-slope value of the best path that reached it.  We
        therefore carry a per-state `cum_path[t, k]` alongside the standard
        Viterbi messages and back-pointers.

        Naming: `log_v` (Viterbi messages) and `delta_i` (random slope) are
        unrelated despite both using the letter delta in HMM literature.
        """
        T = len(y)
        K = self.K
        a0 = ages[0]
        log_pi = np.log(self.pi + 1e-300)
        inv2s = -0.5 / self.sigma2
        log_norm = -0.5 * np.log(2.0 * np.pi * self.sigma2)

        if log_A_all is None:
            log_A_all = self._precompute_log_transitions(ages)

        log_v = np.empty((T, K))            # Viterbi messages, log scale
        psi = np.zeros((T, K), dtype=int)   # back-pointers
        cum_path = np.zeros((T, K))         # cumulative slope of best path → state k

        # -- t = 0: no transition, no cumulative drop yet ---------------
        res0 = y[0] - alpha_i
        emit0 = inv2s * res0 * res0 + log_norm
        log_v[0] = log_pi + emit0

        beta = self.beta  # (K,)
        for t in range(1, T):
            dt = ages[t] - ages[t - 1]
            elapsed = ages[t] - a0

            # For every (predecessor j → current k) pair, extend the cumulative
            # slope by beta[k]*dt and form the candidate prediction + score.
            cum_jk = cum_path[t - 1, :, None] + beta[None, :] * dt        # (K, K)
            pred_jk = alpha_i + delta_i * elapsed + cum_jk                # (K, K)
            res_jk = y[t] - pred_jk
            log_emit_jk = inv2s * res_jk * res_jk + log_norm              # (K, K)
            scores = log_v[t - 1, :, None] + log_A_all[t] + log_emit_jk   # (K, K)

            # Pick the best predecessor for each current state k
            best_j = np.argmax(scores, axis=0)                            # (K,)
            log_v[t] = scores[best_j, np.arange(K)]
            psi[t] = best_j
            cum_path[t] = cum_jk[best_j, np.arange(K)]

        # -- Backtrace ---------------------------------------------------
        states = np.zeros(T, dtype=int)
        states[-1] = int(np.argmax(log_v[-1]))
        for t in range(T - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]

        return states, float(np.max(log_v[-1]))

    # -- Forward-backward (posterior decoding) --------------------------

    def _log_emissions(self, y, ages, alpha_i, delta_i):
        """Pointwise log p(y_t | z_t=k) for each (t, k).

        Note this uses a *pointwise* emission model
            y_t | z_t=k ~ N(alpha_i + (beta_k + delta_i) * (a_t - a_0), sigma2)
        which treats each time point as independent given the current state,
        rather than carrying the cumulative state slope along the path used by
        Viterbi/GPB(1).  It's the standard tractable choice for forward-
        backward and is used here only for post-hoc posterior decoding, never
        during training.
        """
        elapsed = ages - ages[0]
        inv2s = -0.5 / self.sigma2
        log_norm = -0.5 * np.log(2.0 * np.pi * self.sigma2)
        # (T, K) predicted values
        pred = alpha_i + np.outer(elapsed, self.beta + delta_i)
        res = y[:, None] - pred
        return inv2s * res * res + log_norm

    def _precompute_log_transitions(self, ages):
        """Vectorised log A(age) over a whole age grid.

        Returns a (T, K, K) array whose t-th slice is log A(ages[t]).  Cached
        once per EM iteration so the per-individual passes don't recompute it.
        """
        age_n = (ages - self.age_mean) / self.age_sd
        # (T, K, K) logits
        logits = self.W[None, :, :, 0] + age_n[:, None, None] * self.W[None, :, :, 1]
        logits -= logits.max(axis=2, keepdims=True)
        log_norm = np.log(np.exp(logits).sum(axis=2, keepdims=True))
        return logits - log_norm

    def _forward_backward(self, y, ages, alpha_i, delta_i, log_A_all=None):
        """Forward-backward algorithm under the pointwise emission model.

        Returns
        -------
        gamma : (T, K) ndarray
            Posterior marginals gamma[t, k] = P(z_t=k | y_{1:T}).
        ll : float
            Log p(y_{1:T}) under the pointwise emission model.
        """
        T = len(y)
        K = self.K
        log_emit = self._log_emissions(y, ages, alpha_i, delta_i)
        log_pi = np.log(self.pi + 1e-300)

        if log_A_all is None:
            log_A_all = self._precompute_log_transitions(ages)

        # -- Forward --
        log_alpha = np.empty((T, K))
        log_alpha[0] = log_pi + log_emit[0]

        for t in range(1, T):
            # (K, K): log_alpha[t-1, j] + log_A[j, k]  → logsumexp over j
            M = log_alpha[t - 1, :, None] + log_A_all[t]  # (K, K)
            mx = M.max(axis=0)
            log_alpha[t] = mx + np.log(np.exp(M - mx).sum(axis=0)) + log_emit[t]

        # -- Backward --
        log_beta = np.zeros((T, K))

        for t in range(T - 2, -1, -1):
            # log_A_all[t+1][j, k] + log_emit[t+1, k] + log_beta[t+1, k]  → logsumexp over k
            M = log_A_all[t + 1] + log_emit[t + 1] + log_beta[t + 1]  # (K, K)
            mx = M.max(axis=1)
            log_beta[t] = mx + np.log(np.exp(M - mx[:, None]).sum(axis=1))

        # -- Posterior marginals --
        log_gamma = log_alpha + log_beta
        log_gamma -= log_gamma.max(axis=1, keepdims=True)
        gamma = np.exp(log_gamma)
        gamma /= gamma.sum(axis=1, keepdims=True)

        # Log-likelihood from forward pass
        mx = log_alpha[-1].max()
        ll = mx + np.log(np.sum(np.exp(log_alpha[-1] - mx)))

        return gamma, ll

    def _posterior_decode(self, y, ages, alpha_i, delta_i, log_A_all=None):
        """Pointwise MAP decoding: z_t = argmax_k P(z_t=k | y_{1:T})."""
        gamma, ll = self._forward_backward(y, ages, alpha_i, delta_i, log_A_all)
        states = np.argmax(gamma, axis=1)
        return states, ll, gamma

    # -- M-step ----------------------------------------------------------

    def _m_step(self, Y, A_ages, state_seqs, alphas, deltas):
        """Update all model parameters given the current Viterbi paths.

        Six sub-updates, each closed-form (or near-closed-form):
          1. State slopes beta_k from per-segment residual slopes.
          2. Re-label states so beta is descending (canonical order).
          3. Random effects (alpha_i, delta_i) by OLS per individual.
          4. Observation noise sigma2 from full-model residuals.
          5. Age-dependent transition weights W via multinomial logistic
             regression on the realised state transitions.
          6. Initial distribution pi and random-effects hyperparameters
             (mu_re, Sigma_re).
        """
        K = self.K
        N = len(Y)

        # 1. State slopes beta_k --------------------------------------
        # Average per-segment slope (dy/dt - delta_i) within each state.
        diff_sums = np.zeros(K)
        diff_counts = np.zeros(K)
        for y, ages, z, di in zip(Y, A_ages, state_seqs, deltas):
            for t in range(1, len(y)):
                dt = ages[t] - ages[t - 1]
                if dt <= 0:
                    continue
                dy = y[t] - y[t - 1]
                k = z[t]
                diff_sums[k] += dy / dt - di
                diff_counts[k] += 1
        for k in range(K):
            if diff_counts[k] > 0:
                self.beta[k] = diff_sums[k] / diff_counts[k]

        # 2. Canonical state order: beta descending -------------------
        # Permute beta, W, pi and the decoded state sequences accordingly so
        # state 0 is always the shallowest and state K-1 the steepest.
        order = np.argsort(self.beta)[::-1]
        self.beta = self.beta[order]
        self.W = self.W[np.ix_(order, order)]
        self.pi = self.pi[order]
        inv_order = np.argsort(order)
        for i in range(N):
            state_seqs[i] = inv_order[state_seqs[i]]

        # 3. Individual effects (alpha_i, delta_i) via OLS ------------
        # Subtract the path's cumulative state drop, then regress the residual
        # on [1, elapsed] to recover the random intercept and residual slope.
        for i in range(N):
            ages = A_ages[i]
            a0 = ages[0]
            cum = self._cumulative_slopes(ages, state_seqs[i], self.beta)
            resid = Y[i] - cum
            elapsed = ages - a0
            X = np.column_stack([np.ones(len(ages)), elapsed])
            XtX = X.T @ X
            Xty = X.T @ resid
            try:
                coeffs = np.linalg.solve(XtX, Xty)
            except np.linalg.LinAlgError:
                coeffs = np.array([resid.mean(), 0.0])
            alphas[i] = coeffs[0]
            deltas[i] = coeffs[1]

        # 4. Observation noise sigma2 ---------------------------------
        ss = 0.0
        n_obs = 0
        for y, ages, z, ai, di in zip(Y, A_ages, state_seqs, alphas, deltas):
            pred = self._predicted(ages, z, self.beta, ai, di)
            res = y - pred
            ss += np.sum(res ** 2)
            n_obs += len(y)
        self.sigma2 = max(ss / n_obs, 1e-6)

        # 5. Age-dependent transition weights -------------------------
        self._fit_transition_weights(A_ages, state_seqs)

        # 6a. Initial distribution from t=0 state counts (with smoothing)
        init = np.zeros(K)
        for z in state_seqs:
            init[z[0]] += 1
        init += 1e-3
        self.pi = init / init.sum()

        # 6b. Random-effects hyperparameters (sample mean / cov, made PSD)
        re = np.column_stack([alphas, deltas])
        self.mu_re = re.mean(axis=0)
        self.Sigma_re = np.cov(re, rowvar=False)
        self.Sigma_re = 0.5 * (self.Sigma_re + self.Sigma_re.T)
        eigvals = np.linalg.eigvalsh(self.Sigma_re)
        if eigvals.min() < 1e-6:
            self.Sigma_re += np.eye(2) * (1e-6 - eigvals.min())

        return alphas, deltas

    def _fit_transition_weights(self, A_ages, state_seqs):
        """Refit W via per-source multinomial logistic regression.

        For each source state j we have a list of (target_state, age_norm)
        events drawn from all individuals.  The conditional model
            P(z_t = k | z_{t-1} = j, age_t) = softmax_k(W[j, k, 0] + W[j, k, 1] * age_n)
        is fitted by L-BFGS-B with a small L2 penalty on the age coefficient
        to keep them well-conditioned when transitions are rare.
        """
        K = self.K
        if K == 1:
            return

        # Collect every realised transition as parallel arrays
        from_list, to_list, age_list = [], [], []
        for ages, z in zip(A_ages, state_seqs):
            ages_n = (ages - self.age_mean) / self.age_sd
            for t in range(1, len(z)):
                from_list.append(z[t - 1])
                to_list.append(z[t])
                age_list.append(ages_n[t])

        if not from_list:
            return

        from_arr = np.array(from_list)
        to_arr = np.array(to_list)
        age_arr = np.array(age_list)

        # Fit one multinomial logistic regression per source state j
        for j in range(K):
            mask = from_arr == j
            if mask.sum() < 2:
                continue
            targets = to_arr[mask]
            ages_j = age_arr[mask]

            w0 = self.W[j, :, :].ravel().copy()  # (2K,) initial values

            def neg_ll(w_flat):
                w = w_flat.reshape(K, 2)
                logits = w[:, 0] + w[:, 1] * ages_j[:, None]  # (n, K)
                logits -= logits.max(axis=1, keepdims=True)
                log_probs = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
                nll = -log_probs[np.arange(len(targets)), targets].sum()
                # L2 penalty on age coefficients only (intercepts unpenalised)
                nll += 0.01 * np.sum(w[:, 1] ** 2)
                return nll

            result = minimize(neg_ll, w0, method='L-BFGS-B',
                              options={'maxiter': 50, 'ftol': 1e-6})
            self.W[j] = result.x.reshape(K, 2)

    # -- GPB(1) forward pass ---------------------------------------------

    def marginal_log_likelihood(self, Y, A_ages):
        """Sum of per-individual marginal log-likelihoods log p(y | theta).

        Random effects (alpha_i, delta_i) are integrated out analytically by
        the GPB(1) Kalman recursion in `_forward_gpb1`.  Used by `bic` and
        for honest model comparison across K.
        """
        log_A_cache = {}
        total = 0.0
        for y, ages in zip(Y, A_ages):
            cache_key = len(ages)
            if cache_key not in log_A_cache:
                log_A_cache[cache_key] = self._precompute_log_transitions(ages)
            total += self._forward_gpb1(y, ages, log_A_cache[cache_key])
        return total

    def _forward_gpb1(self, y, ages, log_A_all=None):
        """One individual's marginal log-likelihood via GPB(1) + Kalman.

        The exact filter for a switching state-space model has a mixture of
        K^t Gaussians at time t, which is intractable.  Generalised Pseudo
        Bayesian order 1 (GPB(1)) keeps things tractable by collapsing the
        K^2 mixture components arriving at each state to a single Gaussian
        per state at every step (moment-matched).

        Per-state we track:
            log_fw[k]   normalised forward weight log P(z_t=k | y_{1:t})
            ms[k]       posterior mean of [alpha, delta] | y_{1:t}, z_t=k
            Vs[k]       posterior covariance of [alpha, delta] | ..., z_t=k
            cum[k]      cumulative state slope along the path that ends in k

        The observation operator at time t is H_t = [1, elapsed_t]:
            y_t = H_t @ [alpha, delta] + cum + noise.
        """
        T = len(y)
        K = self.K
        a0 = ages[0]

        if log_A_all is None:
            log_A_all = self._precompute_log_transitions(ages)

        # Initialise per-state filter from the random-effects prior
        log_fw = np.zeros(K)
        ms = np.tile(self.mu_re, (K, 1))         # (K, 2)
        Vs = np.tile(self.Sigma_re, (K, 1, 1))   # (K, 2, 2)
        cum = np.zeros(K)                        # cumulative state drop per state
        log_marginal = 0.0

        # -- t = 0: Kalman update with H0 = [1, 0] (elapsed = 0) --------
        H0 = np.array([1.0, 0.0])
        for k in range(K):
            pred = H0 @ ms[k] + cum[k]
            S = H0 @ Vs[k] @ H0 + self.sigma2          # innovation variance
            innov = y[0] - pred
            log_emit = -0.5 * (innov ** 2 / S + np.log(2 * np.pi * S))
            log_fw[k] = np.log(self.pi[k] + 1e-300) + log_emit
            # Kalman gain & posterior update of [alpha, delta]
            K_gain = Vs[k] @ H0 / S
            ms[k] = ms[k] + K_gain * innov
            Vs[k] = Vs[k] - np.outer(K_gain, H0) @ Vs[k]

        # Normalise forward weights, accumulate marginal likelihood
        mx = np.max(log_fw)
        log_c = mx + np.log(np.sum(np.exp(log_fw - mx)))
        log_marginal += log_c
        log_fw -= log_c

        # -- t >= 1: GPB(1) propagate + collapse ------------------------
        for t in range(1, T):
            dt = ages[t] - ages[t - 1]
            elapsed = ages[t] - a0
            H_t = np.array([1.0, elapsed])
            log_A_t = log_A_all[t]

            new_log_fw = np.full(K, -np.inf)
            new_ms = np.zeros((K, 2))
            new_Vs = np.zeros((K, 2, 2))
            new_cum = np.zeros(K)

            for k in range(K):
                # Components arriving at state k from each predecessor j
                log_w = np.empty(K)            # joint log weights P(j, k, y_t | y_{1:t-1})
                tmp_ms = np.empty((K, 2))      # post-update mean per (j → k)
                tmp_Vs = np.empty((K, 2, 2))   # post-update cov  per (j → k)
                tmp_cum = np.empty(K)          # cumulative slope per (j → k)

                for j in range(K):
                    # Extend cumulative slope by current state's contribution
                    c_jk = cum[j] + self.beta[k] * dt
                    pred = H_t @ ms[j] + c_jk
                    S = H_t @ Vs[j] @ H_t + self.sigma2
                    innov = y[t] - pred
                    log_emit = -0.5 * (innov ** 2 / S + np.log(2 * np.pi * S))
                    log_w[j] = log_fw[j] + log_A_t[j, k] + log_emit

                    K_gain = Vs[j] @ H_t / S
                    tmp_ms[j] = ms[j] + K_gain * innov
                    tmp_Vs[j] = Vs[j] - np.outer(K_gain, H_t) @ Vs[j]
                    tmp_cum[j] = c_jk

                # Marginal weight P(z_t=k | y_{1:t-1}) (logsumexp over j)
                mx_w = np.max(log_w)
                new_log_fw[k] = mx_w + np.log(np.sum(np.exp(log_w - mx_w)))

                # Collapse the K mixture components → single Gaussian
                # via moment matching: mean, then law of total covariance.
                weights = np.exp(log_w - new_log_fw[k])
                new_ms[k] = np.einsum('j,jd->d', weights, tmp_ms)
                new_cum[k] = np.dot(weights, tmp_cum)
                diff = tmp_ms - new_ms[k]
                new_Vs[k] = np.einsum('j,jab->ab', weights, tmp_Vs) \
                           + np.einsum('j,ja,jb->ab', weights, diff, diff)

            # Normalise + accumulate
            mx = np.max(new_log_fw)
            log_c = mx + np.log(np.sum(np.exp(new_log_fw - mx)))
            log_marginal += log_c

            log_fw = new_log_fw - log_c
            ms, Vs, cum = new_ms, new_Vs, new_cum

        return log_marginal

    # -- Single EM run -------------------------------------------------

    def _run_em(self, Y, A_ages, alphas, deltas, state_seqs,
                max_iter=80, tol=1e-3, verbose=False, decode='viterbi'):
        """Run EM in place from the current parameter state.

        decode : 'viterbi' or 'posterior'
            'viterbi' (default) uses the cumulative-slope emission and finds
            the best joint state sequence per individual.  'posterior' uses
            the pointwise emission and assigns each time point its marginal
            MAP state.  Posterior decoding mixes badly with the cumulative-
            slope M-step and is mainly useful for post-hoc inspection.

        The reported `total_ll` is the sum of per-individual decoder scores
        (the Viterbi joint log-prob, or the pointwise marginal log-likelihood),
        not the GPB(1) marginal — that's reserved for `marginal_log_likelihood`.
        """
        N = len(Y)
        ll_history = []

        for it in range(max_iter):
            t0 = time()

            total_ll = 0.0
            # Precompute log transitions once per iteration
            log_A_cache = {}
            for i in range(N):
                cache_key = len(A_ages[i])
                if cache_key not in log_A_cache:
                    log_A_cache[cache_key] = self._precompute_log_transitions(A_ages[i])
                log_A_i = log_A_cache[cache_key]
                if decode == 'posterior':
                    z, ll_i, _ = self._posterior_decode(
                        Y[i], A_ages[i], alphas[i], deltas[i], log_A_i)
                else:
                    z, ll_i = self._viterbi(Y[i], A_ages[i], alphas[i], deltas[i], log_A_i)
                state_seqs[i] = z
                total_ll += ll_i
            ll_history.append(total_ll)

            alphas, deltas = self._m_step(Y, A_ages, state_seqs, alphas, deltas)

            elapsed = time() - t0
            if verbose:
                beta_str = ', '.join(f'{b:.4f}' for b in self.beta)
                print(f'  iter {it+1:3d}  LL={total_ll:12.1f}  '
                      f'beta=[{beta_str}]  sigma={np.sqrt(self.sigma2):.3f}  '
                      f'sd_delta={np.sqrt(self.Sigma_re[1,1]):.4f}  ({elapsed:.1f}s)')

            if it > 0 and abs(ll_history[-1] - ll_history[-2]) < tol * abs(ll_history[-2]):
                if verbose:
                    print(f'  Converged at iteration {it+1}.')
                break

        return ll_history, state_seqs, alphas, deltas

    # -- Fit ---------------------------------------------------------------

    def fit(self, Y, A_ages, max_iter=80, tol=1e-3, verbose=False, decode='viterbi'):
        """Initialise from data and run EM until convergence.

        Parameters
        ----------
        Y : list of (T_i,) ndarray
            Per-individual observation sequences.
        A_ages : list of (T_i,) ndarray
            Matching age grids — irregular schedules are supported.
        max_iter, tol, verbose, decode : see `_run_em`.

        Returns
        -------
        ll_history : list of float
            Decoder log-likelihood per EM iteration.
        state_seqs : list of (T_i,) int ndarray
            Final decoded state path per individual.
        alphas : (N,) ndarray
            Posterior point estimates of the random intercept alpha_i.
        deltas : (N,) ndarray
            Posterior point estimates of the random slope delta_i.
        """
        N = len(Y)
        self._initialise(Y, A_ages)

        # Initial guesses for the random effects and state paths
        alphas = np.array([y[0] for y in Y])
        deltas = np.zeros(N)
        state_seqs = [np.zeros(len(y), dtype=int) for y in Y]

        ll_hist, state_seqs, alphas, deltas = self._run_em(
            Y, A_ages, alphas, deltas, state_seqs,
            max_iter=max_iter, tol=tol, verbose=verbose, decode=decode)

        return ll_hist, state_seqs, alphas, deltas

    # -- Model comparison ------------------------------------------------

    def n_parameters(self):
        """Free-parameter count used by BIC.

        Breakdown:
            beta            : K
            W (transitions) : K * K * 2
            pi              : K - 1   (one constraint: rows sum to 1)
            sigma2          : 1
            mu_re           : 2
            Sigma_re        : 3       (symmetric 2x2: 2 variances + 1 covariance)
        """
        K = self.K
        return K + 2 * K * K + (K - 1) + 1 + 2 + 3

    def bic(self, Y, A_ages):
        """Compute BIC = -2 * log p(Y) + n_params * log(n_obs).

        The marginal log-likelihood integrates the random effects analytically
        via GPB(1), so this is an honest cross-K comparison criterion.

        Returns
        -------
        bic : float
            The BIC value (lower = better).
        ll : float
            The marginal log-likelihood that BIC was computed from.
        """
        ll = self.marginal_log_likelihood(Y, A_ages)
        n_obs = sum(len(y) for y in Y)
        return -2 * ll + self.n_parameters() * np.log(n_obs), ll
