# morphological-fusion
Code for morphological fusion study.

## Directory
* `src` contains the code for training/testing a model (either on one language or one feature combination)
* `plots` contains various plots from training (attn and loss)
* `result_plots` contains plots with frequencies and fusions
* `model.py` is the actual python code for running and training one model
* `model_run.py` is the same as `model.py`, but it iterates through all features in a language
* `get_counts.py` is able to compute approximate frequency counts from Wikipedia + Unimorph
* `lat`, `hun`, etc. contain `train` files, `test` files, and `results` files for each feature combination (organized into folders)
* `language_surprisals.txt` contains mean and median surprisal values for each form in a language
