# An Information-Theoretic Characterization of Morphological Fusion
This repo contains code and data for "An information-theoretic characterization of morphological fusion" (at EMNLP 2021).

Contact [neilrathi@gmail.com](mailto:neilrathi@gmail.com) with any questions!

## Directory
* `code` contains the code for creating fusion data for a language, as well as analysis code
* `result_plots` contains the plots used in the paper (main figure, paradigm size vs. fusion, frequency vs. fusion)
* `langdata` has data for
	* fusion by part-of-speech and language
	* paradigm size by part-of-speech and language (vs. fusion)
	* form frequency by feature and language (vs. fusion)

## Requirements
* R. We used version 4.0.3. Analyses and plot generation requires `tidyr`, `dplyr`, `ggplot2`, and `rPref`.
* Python 3.8
* GPU TensorFlow. We used version 2.2.0.
