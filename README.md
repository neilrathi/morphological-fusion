# morphological-fusion
Code for morphological fusion study.

## Directory
* `saved_models` contains the saved model weights for models trained on each feature combination (not in github because files are ~151 MB)
* `model.py` is the actual python code for running and training one model
* `model_run.py` is the same as `model.py`, but it iterates through all features in a language (now works!)
* `lat`, `hun`, etc. contain `train` files, `test` files, and `results` files for each feature combination (organized into folders)
* `language_surprisals.txt` contains mean and median surprisal values for each form in a language
* `plots` contains attention and loss plots for some features
* `full_word_features` and `no_lemma_lat` are old model files

## To-do
* Improve runtime by calling softmax only once at the end of loop

## Completed
* Surprisal of forms
* Reduced runtime w/ early stopping (fixed)
* Created results files
* Model saving
* Fix `FailedPreconditionError` in `model_run.py`
