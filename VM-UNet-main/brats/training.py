"""Training control independent of the CUDA model and data pipeline."""
import math


class EarlyStopping:
    """Count validation checks without a cumulative, significant Dice increase."""

    def __init__(self, enabled=False, patience=10, min_delta=0.0001):
        if not isinstance(enabled, bool):
            raise ValueError("early_stopping.enabled must be a boolean")
        if type(patience) is not int or patience < 1:
            raise ValueError("early_stopping.patience must be a positive integer")
        if (isinstance(min_delta, bool) or not isinstance(min_delta, (int, float))
                or not math.isfinite(min_delta) or min_delta < 0):
            raise ValueError("early_stopping.min_delta must be finite and nonnegative")
        self.enabled, self.patience, self.min_delta = enabled, patience, min_delta
        self.reference_score = None
        self.bad_validations = 0
        self.stopped = False

    def update(self, score):
        if not math.isfinite(score):
            raise ValueError("Validation Mean Dice must be finite")
        if self.reference_score is None or score > self.reference_score + self.min_delta:
            self.reference_score = score
            self.bad_validations = 0
        else:
            self.bad_validations += 1
        self.stopped = self.enabled and self.bad_validations >= self.patience
        return self.stopped

    def state_dict(self):
        return {"reference_score": self.reference_score,
                "bad_validations": self.bad_validations, "stopped": self.stopped}

    def load_state_dict(self, state):
        self.reference_score = state["reference_score"]
        self.bad_validations = state["bad_validations"]
        self.stopped = state["stopped"]
