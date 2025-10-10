
A Python package for DynamicPATCH 
================================================================================================
Dynamic PAtch Transition CHaracterization in explicit space (DynamicPATCH) is a patch-based method that characterizes and quantifies eight types of mutually exclusive and collectively exaustive transition patterns: Appearing, Disappearing, Merging, Splitting, Filling, Perforating, Expanding, and Contracting. DynamicPATCH also computes gross changes in both area and number of patches. See more details about our method in our upcoming manuscript: Zhang et al. (2024) 

# 1. Installation 
----------------------
### Preparation
Before installing the package. Make sure you have Python (3.10 and or above) and pip installed. 


### Install DynamicPATCH
Install the package using the following command:
```
pip install git+https://github.com/zay1996/DynamicPATCH.git@developer-branch
```


# 2. Running the package 
-----------------------
DynamicPATCH provides two ways to run the analysis: the graphical user interface (GUI) option and the command-line option. For ease of use with no coding requirement, use the GUI option. For more flexibility and greater control of the outputs, use the command-line option. Both options require use of a Python interpreter. 

## 2.1 Option one: using the built-in interface
Open `test-interface.py` from the `tests` folder, run the entire script with a Python interpreter. Enter parameters at the pop-up window:

![alt text](img/image.png)

you can save your parameters into a .txt file by clicking on the `Save Parameters` button. Click on `Submit` to upload the parameters. A new window will pop-up:

![alt text](img/image-1.png)

This interface visualizes your input data for each time point. For a test run, use the provided example data under the `exampledata` folder. Click on `Next>>` or `<<Previous` to navigate across maps at different time points. 

Before running the analysis, check what outputs are needed for your analysis, there are two output options: Show Transition Pattern Maps or Show Charts. Choose at least one option. The default will generates both maps and graphical outputs. 

Click on `Run Analysis` and wait until the analysis finishes. The results will be displayed in new tab(s):

![alt text](img/image-2.png)

## 2.2 Option two: command line  
The interface option offers a simple and straightforward way to run DynamicPATCH. However, calling the functions yourself in a Python script or using a command-line interface offers greater flexibility. The script 'test-command.py` gives an example of running DynamicPATCH in a script:

```
import os
import dynamicpatch 
from dynamicpatch import main
package_dir = os.path.dirname(os.path.abspath(dynamicpatch.__file__)) # find directory of the package    
## specify the parameters 
main.run_dynamicpatch(
        workpath = package_dir + '/static/example.xlsx',
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


Note: due to an issue with the current version. Please restart the kernel when changing the input dataset and parameters to avoid errors. 


## Citation
Zhang, A., Pontius Jr, R. G., Bilintoh, T. M., Sangermano, F., & Rogan, J. (2025). DynamicPATCH: Method and software for spatially explicit dynamic patch transition characterization. Landscape Ecology, 40(7), 132. https://doi.org/10.1007/s10980-025-02120-1

