# morpheme-order
Code for morpheme ordering study.

## Directory
* `saved_models` contains the saved model weights for models trained on each feature combination (not in github because files are ~151 MB)
* `model.py` is the actual python code for running and training one model
* `model_run.py` is the same as `model.py`, but it iterates through all features in a language
* `lat`, `hun`, etc. contain `train` files, `test` files, and `results` files for each feature combination (organized into folders)
* `language_surprisals.txt` contains mean and median surprisal values for each form in a language

## To-do:
* Derived forms
* Beam search for top-k
* Fix `FailedPreconditionError` in `model_run.py`

## Completed
* Surprisal of forms
* Reduced runtime w/ early stopping
* Created results files
* Model saving
