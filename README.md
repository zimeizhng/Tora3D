# Tora3D
this is official implement of Tora3D, a deep-learning method for small molecular 3D conformation generation. please read our paper to get more detials.

## setup
you should install pyg in your conda enveriment to run this project

## 预处理
Preprocess the GEOM dataset in the preprocess.ipynb file, where the data_path is the path to the GEOM dataset.（data source：https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/JNGTDF The data file we downloaded is rdkit_folder.tar.gz.）The files generated after preprocessing are placed in ./data_1.

## Intro 
you should run the main_drugs-Copy3_ot-1088.ipynb to get trained model or use our pretraind model to get result.

The small molecule conformations predicted by the model are saved in ./confs_save, where each molecule is a separate folder. The p_*.sdf files are the conformer sets predicted by our model, and the t_*.sdf files are the true conformer sets.

## 可视化
You can check the predicted conformations in the show3Dstructure.ipynb file.
