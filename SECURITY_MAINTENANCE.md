# Dependency security maintenance

Use Python 3.12. The TensorFlow 2.21 / Keras 3.15 runtime replaces the retired 2.8 stack. requirements.in lists direct scientific dependencies; requirements.txt locks and hashes the full graph. Unused legacy TensorBoard, estimator-nightly, Keras-Preprocessing and filesystem plugin pins are removed. The current runtime includes its required components through TensorFlow's own dependencies.

The preprocessing import now uses public Keras layers, and configuration generation/ranking use current pandas concat/items APIs. CI audits the full lockfile and tests initialization, model training/inference, configuration generation and ranking without running external optimization programs. Historical notebook output is retained; full research experiments are not rerun.

Refresh with `uv pip compile --upgrade --universal --python-version 3.12 --generate-hashes requirements.in -o requirements.txt`. Install with `uv pip sync --require-hashes requirements.txt`, audit with `uvx pip-audit -r requirements.txt --disable-pip --no-deps`, and test with `python -m unittest discover -s tests -v` in the project environment.
