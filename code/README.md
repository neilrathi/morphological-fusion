# morphological-fusion/src/
Code is stored here
* `model_run.py` is the Tensorflow model for all features in a language
* `main_fig.R` contains R code for comparing the languages in the main figure
* `plots.R` contains R code for the paradigm size and frequency analyses

To run `model_run.py`, execute the following in the command line:
```
python model_run.py --filepath ../langdata/ --lang lat --lastfeat 'ADJ;ACC;MASC;SG' --embedding 128 --units 512 --batch 512 --learnrate 0.001 --extension .txt
```
where `filepath` is the location of the language folder (i.e. `langdata` should have subfolders `lat`, `hun`, etc. with UniMorph files `lat`, `hun`, etc.). Default hyperparameters are `units = 512`, `embedding_dim = 128`, `BATCH_SIZE = 512`, and `learnrate = 0.001`. The `lastfeat` parameter is optional, and can be used to specify the last feature that was trained (for continuing training); the `extension` parameter is also optional and should be used only if the UniMorph data has a filename extension.

`get_counts.py` is the code to get approximate frequencies from Wikipedia + UniMorph data:
```
python get_counts.py --filepath ../langdata/ --lang tur --langtwo tr
```
where `langtwo` is the two-character language code (if you leave this blank, the two character code is assumed to be the first two of `lang`).