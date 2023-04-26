# Tora3D
this is official implement of Tora3D, a deep-learning method for small molecular 3D conformation generation. please read our paper for more detials.

## Download 
1, After you clone this Repositories in your machine, you should download some files/folders(because the limit of github for big file)
  
  "data_1/rdkit_folder"    
  
  "model_save"
  "prepare_ori_con"       (If you encounter an error message indicating that the file is corrupted)
  "data1/drugs"
  
## setup
you should install pyg in your conda enveriment to run this Repositories

## preprocess
Preprocess the GEOM dataset in the preprocess.ipynb file, where the data_path is the path to the GEOM dataset.（data source：https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/JNGTDF The data file we downloaded is rdkit_folder.tar.gz.）The files generated after preprocessing are placed in ./data_1.

## usage
you should run the main_drugs-Copy3_ot-1088.ipynb to get trained model or use our pretraind model to get result.

The small molecule conformations predicted by the model are saved in ./confs_save, where each molecule is a separate folder. The p_*.sdf files are the conformer sets predicted by our model, and the t_*.sdf files are the true conformer sets.

## visualization
You can check the predicted conformations in the show3Dstructure.ipynb file.
