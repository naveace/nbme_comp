# Overview
This is the main package page for our project.
You probably won't need to interact with anything in this directory outside of `.gitignore` the `src` directory. 

The one crucial thing that you need to do is, once you have your conda/other package manager environment setup, run the following command:
```
pip install -e .
```
Here is an example of what that looks like for me
```
(base) evanvogelbaum@dhcp-10-31-95-23 nbme_comp % conda activate PersonalCoding
(PersonalCoding) evanvogelbaum@dhcp-10-31-95-23 nbme_comp % pip install -e .
Obtaining file:///Users/evanvogelbaum/nbme_comp
  Installing build dependencies ... done
  Getting requirements to build wheel ... done
    Preparing wheel metadata ... done
Installing collected packages: nbme-comp
  Running setup.py develop for nbme-comp
Successfully installed nbme-comp-0.0.post1.dev2+g559fde4
(PersonalCoding) evanvogelbaum@dhcp-10-31-95-23 nbme_comp % 
```

This installs the package into your environment which gets around the awful relative-path-issues that come with python. Now, the following should work assuming you still have your environment activated.
```
(PersonalCoding) evanvogelbaum@dhcp-10-31-95-23 nbme_comp % ipython
Python 3.10.0 (default, Nov 10 2021, 11:24:47) [Clang 12.0.0 ]
Type 'copyright', 'credits' or 'license' for more information
IPython 7.30.1 -- An enhanced Interactive Python. Type '?' for help.

In [1]: from project.hello_world import hello_world

In [2]: hello_world()
Hello World! I am ready to win a Kaggle competition!

In [3]: 
```

The thing I just imported from (hello world) is located in `src/project`. Head there next :) 

# Building

To build this project, run
```
python -m build
```
and find the most recent `.whl` file under `dist/`.