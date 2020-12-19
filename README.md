# morpheme-order
Code for morpheme ordering study.

## Directory
* `saved_models` contains the saved model weights for models trained on each feature combination
* `model` is the actual python code for the model
* `lat`, `hun`, etc. contain `train` files, `test` files, and `results` files for each feature combination (feature combos are organized into folders)

## To-do:
* Derived forms
* Beam search for top-k

## Completed
* Surprisal of forms
* Reduced runtime w/ early stopping
* Created results files
* Model saving
