# morpheme-order
Code for morpheme ordering study.

## Directory
* `saved_models` contains the saved model weights for models trained on each feature combination (not in github because files are ~151 MB)
* `model.py` is the actual python code for running and training one model
* `model_run.py` is the same as `model.py`, but it iterates through all features in a language
* `lat`, `hun`, etc. contain `train` files, `test` files, and `results` files for each feature combination (organized into folders)
* `language_surprisals.txt` contains mean and median surprisal values for each form in a language
* `full_word_features` and `no_lemma_lat` are old model files

## Runtime
For Latin, it takes ~7.5 minutes to create surprisal files for each word within each feature combination for verbs, ~15 for adjectives, and ~30 for nouns. It takes ~2.5 seconds per word to compute surprisal and predict the most likely form. For each feature combination, there are ~2400 forms for verbs, ~4000 for adjectives, and ~8400 for nouns.

## To-do
* ~~Derived forms~~ (unsure if necessary)
* Beam search for top-k

## Completed
* Surprisal of forms
* Reduced runtime w/ early stopping
* Created results files
* Model saving
* Fix `FailedPreconditionError` in `model_run.py`
