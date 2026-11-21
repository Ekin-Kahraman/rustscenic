"""pySCENIC/arboreto with GRNBoost2's original stopping semantics.

scikit-learn < 1.3 defined oob_improvement_[i] on one out-of-bag mask: loss
before round i minus loss after round i, both on round i's own OOB rows.
scikit-learn >= 1.3 subtracts the previous round's OOB loss, which was measured
on a different random mask. arboreto's EarlyStopMonitor averages the last 25
entries, so under >= 1.3 it compares two unrelated 10% subsets.

PairedEarlyStopMonitor recomputes the same-mask improvement from the fitted
stage (real scikit-learn trees, real arboreto code path) and applies arboreto's
unchanged window rule. install() swaps it into arboreto.core.
"""
import numpy as np


class PairedEarlyStopMonitor:
    def __init__(self, window_length=25):
        self.window_length = window_length
        self.history = []

    def __call__(self, current_round, regressor, local):
        oob = ~local['sample_mask']
        y = local['y'][oob]
        weight = local['sample_weight'][oob]
        after = local['raw_predictions'][oob]
        tree = regressor.estimators_[current_round, 0]
        before = after - regressor.learning_rate * tree.predict(local['X'][oob]).reshape(after.shape)
        loss = local['loss_']
        improvement = loss(y, before, weight) - loss(y, after, weight)
        if current_round == 0:
            # Round zero is same-mask in every scikit-learn version: free consistency check.
            assert np.isclose(improvement, regressor.oob_improvement_[0], rtol=1e-6, atol=1e-9)
        self.history.append(improvement)
        if current_round >= self.window_length - 1:
            return np.mean(self.history[-self.window_length:]) < 0
        return False


def install():
    import arboreto.core
    arboreto.core.EarlyStopMonitor = PairedEarlyStopMonitor
