class FailureClassificationModel:
    """
    Classification-based checkpoint decision model.

    Learned decision tree rules from dataset.
    Returns:
    1 → Checkpoint
    0 → Safe
    """

    def predict(self, checkpoint_count, checkpoint_cost):
        # Rule 1
        if checkpoint_count <= 19.5:

            if checkpoint_cost <= 0.0:
                return 0  # Safe

            else:
                return 1  # Checkpoint

        # Rule 2
        else:  # checkpoint_count > 19.5

            if checkpoint_count <= 20.5:
                return 0  # Safe

            else:
                return 0  # Safe

    def should_checkpoint(self, checkpoint_count, checkpoint_cost):
        return self.predict(checkpoint_count, checkpoint_cost) == 1