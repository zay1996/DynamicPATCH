
A Python package for DynamicPATCH
================================================================================================
Dynamic PAtch Transition CHaracterization in explicit space (DynamicPATCH) is a patch-based method that characterizes and quantifies eight types of mutually exclusive and collectively exaustive transition patterns: Appearing, Disappearing, Merging, Splitting, Filling, Perforating, Expanding, and Contracting. DynamicPATCH also computes gross changes in both area and number of patches. See more details about our method in our upcoming manuscript: Zhang et al. (2024) 

# 1. Installation 
----------------------
### Preparation
Before installing the package. Make sure you have Python (3.10 and or above) and pip installed. 

While installing DynamicPATCH will install most of the dependencies, the gdal package need to be installed separately. We recommend installing gdal using conda through the conda-forge channel with the following command:
```
conda install -c conda-forge gdal
```

### Install DynamicPATCH
Install the package using the following command:
```
pip install git+https://github.com/zay1996/DynamicPATCH.git
```


# 2. Running the package 
-----------------------
The current version of DynamicPATCH can be run using command-line with a Python interpreter.

 The script 'test-command.py` gives an example of running DynamicPATCH in a script:

```
from dynamicpatch import main
## specify the parameters 
main.run_dynamicpatch(
        workpath = "DynamicPATCH/exampledata/example.xlsx",
        year = [0,
                1
    ],
        in_nodata = -1, # optional, default = 0
        connectivity = 8, # optional, default = 8
        targ_pre = 1, # optional, default = 1
        study_area = None, # optional, default = None
        map_show = True, # optional, default = True
        chart_show = True, # optional, default = True 
        unit = None, # let program decide automatically
        log_scale = True, # optional
        export_map = False, # optional, default = True
        width = 0.35 # optional, default = 0.35
    )
```

Note: due to an issue with the current version. Please restart the kernel when changing the input dataset and parameters to avoid errors. 


## Citation
Please cite our upcoming publication if you are using DynamicPATCH for your research. 
