import splitfolders
root = "training_set"
splitfolders.ratio(root,output= "Data",
                   ratio = (0.7,0.15,0.15))