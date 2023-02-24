# Tora3D
this is official implement of Tora3D, a deep-learning method for small molecular 3D conformation generation. please read our paper to get more detials.

## setup
you should install pyg in your conda enveriment to run this project

## 预处理
在 preprocess.ipynb 文件中对GEOM数据集进行预处理，其中的data_path为GEOM数据集的路径（数据来源：https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/JNGTDF 我们下载的数据为：rdkit_folder.tar.gz）预处理后产生的文件放在 ./data_1 中。

## Intro 
you should run the main_drugs-Copy3_ot-1088.ipynb to get trained model or use our pretraind model to get result.

模型预测的小分子构象保存在 ./confs_save 中，每一个分子为一个文件夹，其中p_*.sdf是我们的模型预测的构象集， t_*.sdf是真实的构象集。

## 可视化
你可以在 show3Dstructure.ipynb 中查看预测的构象 



