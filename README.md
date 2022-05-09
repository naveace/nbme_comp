# NBME Kaggle Competition
Welcome to our project! This set of directories contains our active work. It is also a package with can be built using `python -m build` from `nbme_comp/`. We developed here and then built our project and extracted a `.whl` file to use in our Kaggle submissions. The competition webpage is available [here](https://www.kaggle.com/competitions/nbme-score-clinical-patient-notes). 


# Overview
`src/project/` contains all of our code

`src/project/data/` contains data from the competition as well as cleaning scripts

`src/project/embedding/` is a deprecated attempt to build a large-scale embedding system (it was too challenging to use supercloud given environment constraints on that HPC)

`src/project/experiments/` contains our main work: data cleaning and exploration notebooks, our main models, etc...

`src/project/utils` contains various utils
