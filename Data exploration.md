# Data Preparation for training
The dataset contains training and test data folders which includes both images and videos
In here i train the model using labeled images given in the source dataset
The annotations are given in the COCO format as a JSON file. for the YOLO training we need to convert these annotations to YOLO compatible format. In the following codes there are stepwise approach to do the data exploration and the annotation conversion as well as the train and validation splits

The data training was done from DTU HPC server GPU's

## DTU HPC 
```bibtex
@misc{DTU_DCC_resource,
    author    = {{DTU Computing Center}},
    title     = {{DTU Computing Center resources}},
    year      = {2024},
    publisher = {Technical University of Denmark},
    doi       = {10.48714/DTU.HPC.0001},
    url       = {https://doi.org/10.48714/DTU.HPC.0001},
}
```

##loading the annotation file
```python
from pycocotools.coco import COCO

annotation = path/trainImages.json"
coco = COCO(annotation)
```
