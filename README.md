# CE888 Stress Detection: Nurses

In order to extract the dataset, you need to download and extract the datasets1.zip, datasets2.zip and datasets3.zip as 
well as the combined_fullpreprocessed.zip. This will extract the preprocessed data that can be used to build the models.

If you require the full dataset then it can be downloaded from, run python loader.py to extract from the dataset:
https://datadryad.org/stash/dataset/doi:10.5061/dryad.5hqbzkh6f

In order to run feature extraction, you can use the command python featureextraction.py, provided that you already have 
the dataset described above set up.

Once you have extracted the features, you can use the MakeModel class in modelmaker.py to generate the models.

notebook.ipynb contains some data exploration and visualisation as well as the 5-fold cross validation for each model. 


