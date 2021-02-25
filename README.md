# morphological-fusion
Code for morphological fusion study.

## Directory
* `src` contains the code for training/testing a model (either on one language or one feature combination)
* `plots` contains various plots (attention and loss) from training
* `model.py` is the actual python code for running and training one model
* `model_run.py` is the same as `model.py`, but it iterates through all features in a language
* `lat`, `hun`, etc. contain `train` files, `test` files, and `results` files for each feature combination (organized into folders)
* `language_surprisals.txt` contains mean and median surprisal values for each form in a language

## To-do
* Run code on multiple languages

## Completed
* Surprisal of forms
* Reduced runtime w/ early stopping (fixed)
* Created results files
* Model saving
* Fix `FailedPreconditionError` in `model_run.py`
