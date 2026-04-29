class FeatureExtractor:
    """
    EXTENDED: Extracts decision features for adaptive checkpoint placement.
    Now includes dynamic state and risk-weighted metrics to help ML outperform Analytical.
    """

    def __init__(self, structural_metrics):
        # Ensure we have a dictionary to avoid AttributeErrors
        self.structural_metrics = structural_metrics if structural_metrics else {}

    def extract(
        self,
        work_since_last_checkpoint,
        failure_rate,
        avg_block_cost,
        dynamic_state_size,
        execution_variance=0.0
    ):
        # 1. Base Structural Metrics
        total_blocks = max(self.structural_metrics.get("total_basic_blocks", 1), 1)
        loop_count = self.structural_metrics.get("loop_count", 0)
        branch_count = self.structural_metrics.get("branch_count", 0)
        cyclomatic_complexity = self.structural_metrics.get("cyclomatic_complexity", 0)

        # 2. Normalized Densities
        loop_density = loop_count / total_blocks
        branch_density = branch_count / total_blocks

        # 3. Temporal Risk (How much "pain" is accumulated)
        # Scaled to keep the number within a reasonable range for the ML weights
        work_risk_factor = work_since_last_checkpoint * failure_rate

        # 4. State Pressure (Relative cost of saving right now)
        # We normalize this by the total blocks to see if the state is "heavy"
        # relative to the program size.
        state_pressure = dynamic_state_size / total_blocks

        # 5. NEW: Efficiency Ratio (Work Done vs. State to Save)
        # High ratio = It's a "profitable" time to save.
        # Low ratio = State is too big relative to the work we've done.
        efficiency_ratio = work_since_last_checkpoint / (dynamic_state_size * 0.001 + 1e-9)

        features = {
            "work_since_last_checkpoint": work_since_last_checkpoint,
            "failure_rate": failure_rate,
            "avg_block_cost": avg_block_cost,
            "execution_variance": execution_variance,
            "loop_density": loop_density,
            "branch_density": branch_density,
            "cyclomatic_complexity": cyclomatic_complexity,
            "state_pressure": state_pressure,
            "work_risk_factor": work_risk_factor,
            "efficiency_ratio": efficiency_ratio
        }

        return features