# Tora3D
this is official implement of Tora3D, a deep-learning method for small molecular 3D conformation generation. please read our paper for more detials.

## Download 
After you clone this Repositories in your machine, you should download some files/folders(because the limit of github for big file)
  
1, "data_1/rdkit_folder"   download rdkit_folder.tar.gz file from https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/JNGTDF, put it in data_1 floder, then unzip it use tar commond. The file structure after unzip is as follows:
  
     Tora3D
     -data_1
     --rdkit_folder
     ---druds
     ---qm9
2,
  "model_save"         
  "prepare_ori_con"       (If you encounter an error message indicating that the file is corrupted)
  "data1/drugs"
  These three files are obtained from Kuaipan:
        https://pan.quark.cn/s/c32e62fe57b0
        Extraction code: nKrC
  
## setup
you should install pyg in your conda enveriment to run this Repositories

## preprocess
Preprocess the GEOM dataset in the preprocess.ipynb file, where the data_path is the path to the GEOM dataset.（data source：https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/JNGTDF The data file we downloaded is rdkit_folder.tar.gz.）The files generated after preprocessing are placed in ./data_1/drugs/.

## usage
you should run the main_drugs-Copy3_ot-1088.ipynb to get trained model or use our pretraind model to get result.

The small molecule conformations predicted by the model are saved in ./confs_save, where each molecule is a separate folder. The p_*.sdf files are the conformer sets predicted by our model, and the t_*.sdf files are the true conformer sets.

## visualization
You can check the predicted conformations in the show3Dstructure.ipynb file.
