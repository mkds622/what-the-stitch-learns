"""Sparse cross-coders for comparing representations in vision models.

The package implements the experiment described in PROJECT.md: a sparse coder is
placed at a cut point in a frozen vision transformer and trained under different
objectives, and the dictionaries it learns are compared across those objectives.

Module map
----------
config         every setting that distinguishes one run from another
logging        experiment tracking, process logging and crash capture
models         loading, tracing and cutting the backbones          
sweep          locating depths where a stitch works       
coder          the sparse coder                                    
objectives     what the coder is trained to reproduce              
activations    supplying the coder, cached or computed             
train          the training loop                                   
metrics        Tier 0 validity gates and in-place evaluation       

"""

__version__ = "0.1.0"
