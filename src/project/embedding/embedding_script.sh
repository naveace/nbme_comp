#!/bin/bash
# Takes as kwarg the embedding model we want to use as the Hugging Face class name: `Model Name`
# Embeds all patient notes in the data directory with sub-directory `Model Name` and save each of them in a file `pn_id`.npy 
#   where pn_id is the id of the patient note
#   ex: outputs to `data/embeddings/all-mpnet-base-v2/2.npy`
# TODO: Would like this to be runnable in triples mode on supercloud and also on local machine for testing. Please see https://supercloud.mit.edu/job-arrays-llsub-triples-3-steps

python embedding_script.py $1