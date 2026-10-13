PY ?= python

.PHONY: install test train tournament figures all
install:
	$(PY) -m pip install -r requirements.txt
test:
	$(PY) -m pytest -q
train:
	$(PY) scripts/train_rl.py --iters 1000
tournament:
	$(PY) scripts/run_tournament.py --races 16 --workers 4
figures:
	$(PY) scripts/make_figures.py
all: train tournament figures
