
A Python package for DynamicPATCH 
================================================================================================
Spatially explicit Dynamic PAtch Transition CHaracterization (DynamicPATCH) is a patch-based method that characterizes and quantifies eight types of mutually exclusive and collectively exaustive transition patterns: Appearing, Disappearing, Merging, Splitting, Filling, Perforating, Expanding, and Contracting. DynamicPATCH also computes gross changes in both area and number of patches. See more details about our method in our paper: Zhang, A., Pontius Jr, R. G., Bilintoh, T. M., Sangermano, F., & Rogan, J. (2025). DynamicPATCH: Method and software for spatially explicit dynamic patch transition characterization. Landscape Ecology, 40(7), 132. https://doi.org/10.1007/s10980-025-02120-1

# Installation 
----------------------
### Preparation
Before installing the package. Make sure you have Python (3.10 and or above) and pip installed. 


### Install DynamicPATCH
Install the package using the following command:
```
pip install git+https://github.com/zay1996/DynamicPATCH.git@developer-branch
```


# Running the package 
-----------------------
DynamicPATCH provides two ways to run the analysis: the graphical user interface (GUI) option and the command-line option. For ease of use with no coding requirement, use the GUI option. For more flexibility and greater control of the outputs, use the command-line option. Both options require use of a Python interpreter. 

## Example code 
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
                1],
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

## List of arguments 
```python

----------
workpath : str
    Path to the input data. This can be either a folder containing the input maps
    or the path to a single input file.
    Example: "C:/Users/Analysis/piemarsh.tif"

year : list of int
    Years corresponding to the input maps, provided in temporal order.
    Example: [1938, 1971, 2013]

in_nodata : int, default=0
    NoData value in the input maps.

connectivity : int, default=8
    Connectivity rule used to identify patches. Options are 4 and 8.
    Use 4 for rook connectivity and 8 for queen connectivity.

targ_pre : int, default=1
    Pixel value representing the presence category of interest.

study_area : str, optional
    Name of the study area. This name is used in output labels and/or file names.
    Default is None.

map_show : bool, default=True
    Whether to display the result map.

chart_show : bool, default=True
    Whether to generate result graphics.

export_map : str or None, default=None Output path prefix for exporting trajectory maps. 
The path should include the directory and output file name, but not the ".tif" extension. 
If None, maps are not exported. Example: export_map = r"D:\analysis\dynamicpatch_map"

unit : str, default="Default"
    Area unit used in the result graphics. Options are "pixels", "sqm2", "km2",
    and "Default". If set to "Default", the program automatically selects an
    appropriate area unit based on the size of the input data.

log_scale : bool, default=True
    Whether to display the size distribution graph on a log scale. In the current
    version, the size distribution graph is displayed only on a log scale.

width : float, default=0.35
    Width of the bars in the bar charts.

rotation : int, default=0
    Rotation angle of the year labels in the stacked bar chart.

res : int or float, optional
    Spatial resolution of the input maps in meters. If not specified, the
    resolution is read automatically from the input map.
```


Note: due to an issue with the current version. Please restart the kernel when changing the input dataset and parameters to avoid errors. 

## Outputs 

The function returns a dictionary containing the transition pattern map and summary tables:

```python
outputs = {
    "pattern": pattern,
    "patch_size": df_patch_size,
    "patch_num": df_patch_num,
    "gainloss": df_gainloss_all,
    "increase_decrease": df_inde_all,
}
```

Access each output by its key:

```python

pattern = outputs["pattern"]
df_patch_size = outputs["patch_size"]
df_patch_num = outputs["patch_num"]
df_gainloss_all = outputs["gainloss"]
df_inde_all = outputs["increase_decrease"]
```

### `pattern`

A 3-dimensional array containing the transition pattern maps for all time intervals. Each layer represents the spatial transition pattern between two consecutive input maps. For example, if the input maps correspond to `[1938, 1971, 2013]`, the output contains two transition maps: one for 1938–1971 and one for 1971–2013.

### `patch_size`

A pandas DataFrame containing patch size information for each transition type and each time interval. 

### `patch_num`

A pandas DataFrame containing the number of patches for each transition type and each time interval. 

### `gainloss`

A pandas DataFrame summarizing the annual size of each transition type.

### `increase_decrease`

A pandas DataFrame summarizing the annual number of increase and decrease patches contributed by each transition type. 


## Citation
Zhang, A., Pontius Jr, R. G., Bilintoh, T. M., Sangermano, F., & Rogan, J. (2025). DynamicPATCH: Method and software for spatially explicit dynamic patch transition characterization. Landscape Ecology, 40(7), 132. https://doi.org/10.1007/s10980-025-02120-1

