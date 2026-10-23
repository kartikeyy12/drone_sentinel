VENV ?= .venv
PY   := $(VENV)/bin/python
PIP  := $(VENV)/bin/pip

.PHONY: bench setup check data manifest features m2 train ablation robustness final stream-check scenarios eval-fa range report test

setup:
	python3.11 -m venv $(VENV)
	$(PIP) install -U pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

check:
	$(PY) scripts/check_env.py

data:
	$(PY) scripts/download_data.py --only all
	$(MAKE) manifest

manifest:
	$(PY) scripts/build_manifest.py

features:
	$(PY) scripts/extract_features.py
	$(PY) scripts/plot_denoise_examples.py

m2:
	$(PY) scripts/svd_denoise_experiment.py
	$(PY) scripts/pca_variance.py

train:
	$(PY) scripts/train_cv.py
ablation:
	$(PY) scripts/run_ablation.py
robustness:
	$(PY) scripts/run_robustness.py
final:
	$(PY) scripts/build_final.py
stream-check:
	$(PY) scripts/check_streaming.py
scenarios:
	$(PY) scripts/make_scenarios.py
eval-fa: scenarios
	$(PY) scripts/eval_false_alarms.py
range:
	$(PY) scripts/run_range.py
bench:
	$(PY) scripts/bench_distance.py
report:
	$(PY) scripts/make_report_notes.py
	$(PY) scripts/make_report.py

test:
	$(PY) -m pytest -q
