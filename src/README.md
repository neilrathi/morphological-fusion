# morphological-fusion/src/
Code for the model is stored here
* `model.py` is the Tensorflow model for one set of trainfeats
* `model_run.py` is the Tensorflow model for all features in a language
* `model_pytorch.py` is the Pytorch model (WIP)

To run `model_run.py`, execute the following in the command line:
```
python model_run.py --filepath ../langdata/ --lang lat --embedding 128 --units 512 --batch 512 --learnrate 0.001
```
where `filepath` is the location of the language folder (i.e. `langdata` should have subfolders `lat`, `hun`, etc. with Unimorph files `lat`, `hun`, etc.). Default hyperparameters are `units = 512`, `embedding_dim = 128`, `BATCH_SIZE = 512`, and `learnrate = 0.001`.
