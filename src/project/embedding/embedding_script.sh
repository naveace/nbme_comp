#!/bin/bash
# Should take as kwarg the embedding model we want to use as the Hugging Face class name: `Model Name`
# Should embed all patient notes in a directory with name `Model Name` and save each of them in a file `pn_id`.npy 
#   where pn_id is the id of the patient note
# TODO: Would like this to be runnable in triples mode on supercloud and also on local machine for testing. Please see https://supercloud.mit.edu/job-arrays-llsub-triples-3-steps

python embedding_script.py $1