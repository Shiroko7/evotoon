import os
import unittest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

import numpy as np
import pandas as pd

import evotoon
from data_classes import IntParam


class CompatibilityTests(unittest.TestCase):
    def test_initial_population_keeps_parameter_bounds(self):
        evotoon.make_seed(42)
        population = evotoon.initialization(
            4, int_parameters_list=[IntParam("x", 1, 8)]
        )
        self.assertEqual(len(population), 4)
        self.assertTrue(all(1 <= row["x"] <= 8 for row in population))

    def test_model_trains_and_predicts_with_modern_keras(self):
        features = np.array([[1, 2], [2, 3], [3, 4], [4, 5]], dtype=np.float32)
        model = evotoon.create_model(features, layers=1, neurons=[4])
        loss = model.train_on_batch(features, np.arange(4, dtype=np.float32))
        predictions = model(features, training=False).numpy()
        self.assertTrue(np.isfinite(loss))
        self.assertEqual(predictions.shape, (4, 1))
        self.assertTrue(np.isfinite(predictions).all())

    def test_configuration_generation_preserves_rows_with_pandas_two(self):
        class Predictor:
            def predict(self, values):
                return np.zeros((len(values), 1), dtype=np.float32)

        batch = pd.DataFrame({"x": [1, 2], "Step_Found": [0, 0]})
        generated = evotoon.generate_configurations(
            Predictor(), batch, {}, hc_max_tries=0, index_start=4
        )
        self.assertEqual(generated["x"].tolist(), [1, 2])
        self.assertEqual(generated.index.tolist(), [4, 5])

    def test_ranking_uses_current_pandas_iteration_api(self):
        instances = ["a", "b", "c"]
        old = {0: pd.DataFrame({"instance_name": instances, "score": [1, 1, 1]})}
        new = {1: pd.DataFrame({"instance_name": instances, "score": [2, 2, 2]})}
        batch, evaluations = evotoon.select_configurations_by_ranking(
            pd.DataFrame({"x": [1]}), old,
            pd.DataFrame({"x": [2]}, index=[1]), new,
            instances, population_size=1, friedman_test=False,
        )
        self.assertEqual(batch["x"].tolist(), [2])
        self.assertEqual(evaluations[0]["score"].mean(), 2)


if __name__ == "__main__":
    unittest.main()
