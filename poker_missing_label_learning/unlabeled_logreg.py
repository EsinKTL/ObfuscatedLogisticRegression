"""
    Logistic regression with missing labels (X, y_obs).

    y_obs = -1   missing label (unlabeled)
    y_obs = 0/1  observed label (labeled)

    Parameters:
    
    method : str
        'label_propagation' or 'em'
    lambdas : array-like or None
        Lambda grid for FISTA. If None, uses logspace(-4, 1, 15).
    measure : str
        Metric for lambda selection. Default 'f1'.
    max_iter_em : int
        Maximum number of iterations for EM.
    max_iter_fista : int
        Maximum number of iterations for FISTA.
    feature_idx : int
        Feature index for MAR1 and MNAR. Default 11 (pot_growth).
    mnar_y_weight : float
        Weight of the label in MNAR. Default 5.0.
    c_values : list
        c values for run_mcar_sensitivity().
    random_state : int or None
"""
def __init__(
        self,
        method='label_propagation',
        lambdas=None,
        measure='f1',
        max_iter_em=10,
        max_iter_fista=500,
        feature_idx=11,
        mnar_y_weight=5.0,
        c_values=None,
        random_state=42,
    ):
        if method not in ('label_propagation', 'em'):
            raise ValueError(
                f"Unknown method '{method}'. "
                "Choose 'label_propagation' or 'em'."
            )
        self.method         = method
        self.lambdas        = lambdas if lambdas is not None else np.logspace(-4, 1, 15)
        self.measure        = measure
        self.max_iter_em    = max_iter_em
        self.max_iter_fista = max_iter_fista
        self.feature_idx    = feature_idx
        self.mnar_y_weight  = mnar_y_weight
        self.c_values       = c_values if c_values is not None else [0.1, 0.2, 0.3, 0.4, 0.5]
        self.random_state   = random_state

        # Trained selectors — filled after respective fit call
        self._selector_unlabeled = None
        self._selector_naive     = None
        self._selector_oracle    = None


    def fit(self, X, y_obs, X_valid, y_valid):
        """
        Train model with labeled + unlabeled data.

        Uses self.method ('label_propagation' or 'em') for Y-completion.
        Train FISTA with completed Y.

        Parameters
        ----------
        X : array (n_samples, n_features)
        y_obs : array (n_samples,) — -1 missing label
        X_valid, y_valid : validation set (fully labeled)
        """
        X     = np.asarray(X, dtype=float)
        y_obs = np.asarray(y_obs, dtype=float)

        if self.method == 'label_propagation':
            y_complete = self._label_propagation(X, y_obs, X_valid, y_valid)
        else:
            y_complete = self._em(X, y_obs, X_valid, y_valid)

        self._selector_unlabeled = self._train_fista(X, y_complete, X_valid, y_valid)
        return self

    def naive_fit(self, X, y_obs, X_valid, y_valid):
        """
        Train model using only labeled observations (y_obs != -1).

        Completely ignores unlabeled observations.
        This is the Naive baseline in Task 3.

        Parameters
        ----------
        X : array (n_samples, n_features)
        y_obs : array (n_samples,) — -1 missing label
        X_valid, y_valid : validation set
        """
        X     = np.asarray(X, dtype=float)
        y_obs = np.asarray(y_obs, dtype=float)

        mask = (y_obs != -1)
        self._selector_naive = self._train_fista(
            X[mask], y_obs[mask], X_valid, y_valid
        )
        return self

    def oracle_fit(self, X, y_true, X_valid, y_valid):
        """
        Train model with all true labels (reference benchmark).

        Assumes that even the missing labels are known.
        This is the Oracle method in Task 3 — the achievable upper bound.

        Parameters
        ----------
        X : array (n_samples, n_features)
        y_true : array (n_samples,) — all true labels
        X_valid, y_valid : validation set
        """
        X      = np.asarray(X, dtype=float)
        y_true = np.asarray(y_true, dtype=float)

        self._selector_oracle = self._train_fista(X, y_true, X_valid, y_valid)
        return self

    def compare(self, X_test, y_test, print_results=True):
        """
        Compare results of fit(), naive_fit(), oracle_fit() on test set.

        Only those that have been called appear. If not called, that column is missing.

        Parameters
        ----------
        X_test : array
        y_test : array
        print_results : bool

        Returns
        -------
        results : dict[str, dict[str, float]]
        """
        results = {}

        if self._selector_naive is not None:
            results['Naive'] = self._metrics(self._selector_naive, X_test, y_test)

        if self._selector_unlabeled is not None:
            label = 'LabelProp' if self.method == 'label_propagation' else 'EM'
            results[label] = self._metrics(self._selector_unlabeled, X_test, y_test)

        if self._selector_oracle is not None:
            results['Oracle'] = self._metrics(self._selector_oracle, X_test, y_test)

        if print_results:
            self._print_table("Comparison", results)

        return results
      
    # PUBLIC: run_schemes

    def run_schemes(
        self, X, y,
        X_valid, y_valid,
        X_test, y_test,
        c=0.3,
        schemes=('mcar', 'mar1', 'mar2', 'mnar'),
        save_plot=True,
        plot_filename='results_schemes.png',
    ):
        scheme_kwargs = {
            'mcar': {'scheme': 'mcar'},
            'mar1': {'scheme': 'mar1', 'feature_idx': self.feature_idx},
            'mar2': {'scheme': 'mar2'},
            'mnar': {'scheme': 'mnar', 'feature_idx': self.feature_idx,
                     'y_weight': self.mnar_y_weight},
        }

        all_results = {}

        print("=" * 60)
        print(f"  EXPERIMENT 1: Comparison of 4 Schemes  (c={c})")
        print("=" * 60)

        for scheme in schemes:
            print(f"\n--- {scheme.upper()} ---")

            y_obs = generate_missing(
                X, y, c=c,
                random_state=self.random_state,
                **scheme_kwargs[scheme]
            )
            n_miss = (y_obs == -1).sum()
            print(f"Missing labels: {n_miss} ({n_miss / len(y_obs) * 100:.1f}%)")

            self.naive_fit(X, y_obs, X_valid, y_valid)
            self.fit(X, y_obs, X_valid, y_valid)
            self.oracle_fit(X, y, X_valid, y_valid)

            results = self.compare(X_test, y_test, print_results=True)
            all_results[scheme] = results

        if save_plot:
            self._plot_schemes(all_results, list(schemes), plot_filename)

        return all_results

    # PUBLIC: run_mcar_sensitivity

    def run_mcar_sensitivity(
        self, X, y,
        X_valid, y_valid,
        X_test, y_test,
        save_plot=True,
        plot_filename='results_mcar_c.png',
    ):
        """
        Analyzes the effect of different c values under MCAR.

        For each c in self.c_values, runs Naive, fit(), Oracle and records 4 metrics.

        Parameters
        ----------
        X, y : full training data
        X_valid, y_valid : validation set
        X_test, y_test : test set
        save_plot : bool
        plot_filename : str

        Returns
        -------
        mcar_results : dict[metric, dict[method, list[float]]]
        """
        metrics_list = ['accuracy', 'balanced_accuracy', 'f1', 'roc_auc']
        method_label = 'LabelProp' if self.method == 'label_propagation' else 'EM'
        methods      = ['Naive', method_label, 'Oracle']

        mcar_results = {
            m: {method: [] for method in methods}
            for m in metrics_list
        }

        print("\n" + "=" * 60)
        print("  EXPERIMENT 2: MCAR — Different c Values")
        print("=" * 60)

        for c in self.c_values:
            print(f"\n  c = {c}")
            y_obs = generate_missing(
                X, y, scheme='mcar', c=c, random_state=self.random_state
            )

            self.naive_fit(X, y_obs, X_valid, y_valid)
            self.fit(X, y_obs, X_valid, y_valid)
            self.oracle_fit(X, y, X_valid, y_valid)

            r = self.compare(X_test, y_test, print_results=False)

            for metric in metrics_list:
                mcar_results[metric]['Naive'].append(r['Naive'][metric])
                mcar_results[metric][method_label].append(r[method_label][metric])
                mcar_results[metric]['Oracle'].append(r['Oracle'][metric])

        if save_plot:
            self._plot_mcar_sensitivity(mcar_results, methods, plot_filename)

        return mcar_results

    # PUBLIC: predict

    def predict_proba(self, X):
        """Return probabilities from model trained with fit()."""
        if self._selector_unlabeled is None:
            raise RuntimeError("fit() must be called first.")
        return self._selector_unlabeled.predict_proba(X)

    def predict(self, X, threshold=0.5):
        """Return binary predictions from model trained with fit()."""
        return (self.predict_proba(X) >= threshold).astype(int)

    # PRIVATE: helpers

    def _train_fista(self, X_tr, y_tr, X_val, y_val):
        """Train a new FISTASelector and return it."""
        sel = FISTASelector(
            lambdas=self.lambdas,
            max_iter=self.max_iter_fista,
        )
        sel.fit(X_tr, y_tr, X_val, y_val, measure=self.measure)
        return sel

    def _metrics(self, selector, X_test, y_test):
        """Compute 4 metrics for a FISTASelector and return as dict."""
        proba = selector.predict_proba(X_test)
        pred  = (proba >= 0.5).astype(int)
        return {
            'accuracy':          accuracy_score(y_test, pred),
            'balanced_accuracy': balanced_accuracy_score(y_test, pred),
            'f1':                f1_score(y_test, pred, zero_division=0),
            'roc_auc':           roc_auc_score(y_test, proba),
        }

    # PRIVATE: Y-completion algorithms

    def _label_propagation(self, X, y_obs, X_valid, y_valid):
        """
        Algorithm 1: Label Propagation.

        Step 1 — Train initial model on labeled observations.
        Step 2 — Predict labels for unlabeled observations using this model.
        Step 3 — Return completed Y array.

        Single round. Errors in the initial model are not corrected.
        """
        labeled_mask   = (y_obs != -1)
        unlabeled_mask = (y_obs == -1)

        init_sel = self._train_fista(
            X[labeled_mask], y_obs[labeled_mask], X_valid, y_valid
        )

        y_complete = y_obs.copy()
        if unlabeled_mask.sum() > 0:
            proba = init_sel.predict_proba(X[unlabeled_mask])
            y_complete[unlabeled_mask] = (proba >= 0.5).astype(float)

        return y_complete

    def _em(self, X, y_obs, X_valid, y_valid):
        """
        Algorithm 2: Expectation-Maximization.

        Initialization: first estimate using label propagation.
        Each iteration:
          M-step — train model with current labels.
          E-step — update labels for unlabeled observations.
        Continue until labels stop changing.

        Performance may degrade in MNAR due to confirmation bias.
        """
        unlabeled_mask = (y_obs == -1)

        y_current = self._label_propagation(X, y_obs, X_valid, y_valid)

        for iteration in range(self.max_iter_em):

            sel = self._train_fista(X, y_current, X_valid, y_valid)

            if unlabeled_mask.sum() == 0:
                break

            proba = sel.predict_proba(X[unlabeled_mask])
            y_new = y_current.copy()
            y_new[unlabeled_mask] = (proba >= 0.5).astype(float)

            n_changed = (y_new[unlabeled_mask] != y_current[unlabeled_mask]).sum()
            y_current = y_new

            if n_changed == 0:
                print(f"  EM converged at iteration {iteration + 1}.")
                break

        return y_current

    # PRIVATE: printing & plotting
    @staticmethod
    def _print_table(title, results):
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")
        print(f"{'Method':18s}  {'Acc':>7}  {'BalAcc':>7}  {'F1':>7}  {'AUC':>7}")
        print(f"{'-'*60}")
        for name, m in results.items():
            print(
                f"{name:18s}  "
                f"{m['accuracy']:.4f}   "
                f"{m['balanced_accuracy']:.4f}   "
                f"{m['f1']:.4f}   "
                f"{m['roc_auc']:.4f}"
            )

    @staticmethod
    def _plot_schemes(all_results, schemes, filename):
        metrics = ['accuracy', 'balanced_accuracy', 'f1', 'roc_auc']
        methods = list(next(iter(all_results.values())).keys())
        colors  = ['#E24B4A', '#378ADD', '#1D9E75', '#888780']
        x       = np.arange(len(schemes))
        width   = 0.18

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            for i, (method, color) in enumerate(zip(methods, colors)):
                vals = [all_results[s][method][metric] for s in schemes]
                ax.bar(
                    x + i * width - (len(methods) - 1) / 2 * width,
                    vals, width, label=method, color=color, alpha=0.85
                )
            ax.set_xticks(x)
            ax.set_xticklabels([s.upper() for s in schemes])
            ax.set_title(metric)
            ax.set_ylim(0, 1.05)
            ax.legend(fontsize=8)
            ax.grid(axis='y', alpha=0.3)
            ax.set_ylabel('score')

        fig.suptitle('Method Comparison — 4 Missing Data Schemes', fontsize=14)
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"Plot saved: {filename}")

    def _plot_mcar_sensitivity(self, mcar_results, methods, filename):
        metrics = ['accuracy', 'balanced_accuracy', 'f1', 'roc_auc']
        colors  = ['#E24B4A', '#378ADD', '#888780']

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            for method, color in zip(methods, colors):
                ax.plot(
                    self.c_values, mcar_results[metric][method],
                    'o-', label=method, color=color, linewidth=2, markersize=6
                )
            ax.set_xlabel('c (missing ratio)')
            ax.set_ylabel('score')
            ax.set_title(metric)
            ax.set_ylim(0, 1.05)
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

        fig.suptitle('MCAR — Different c Values', fontsize=14)
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"Plot saved: {filename}")
