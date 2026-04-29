class HybridCheckpointModel:
    """
    Dual-engine decision model.
    Combines Risk Classification with Efficiency Regression.
    """

    def __init__(self, regression_model, classification_model, threshold=0.85):
        self.reg_model = regression_model
        self.clf_model = classification_model
        self.threshold = threshold

    def should_checkpoint(self, features: dict):
        # 1. Classification: Is this a 'High Risk' instruction? (Binary)
        # Based on structural complexity and failure rate.
        is_risky = self.clf_model.predict(features)

        # 2. Regression: What is the predicted efficiency if we save NOW? (0.0 - 1.0)
        # Based on state_pressure and work_since_last_checkpoint.
        predicted_efficiency = self.reg_model.predict(features)

        # 3. Hybrid Logic:
        # We only save if the risk is high AND the efficiency is above our target.
        # This prevents 'Panic Checkpointing' that destroys performance.
        if is_risky == 1 and predicted_efficiency >= self.threshold:
            return True

        return False