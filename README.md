# Tora3D
this is official implement of Tora3D, a deep-learning method for small molecular 3D conformation generation. please read our paper to get more detials.

## setup
you should install pyg in your conda enveriment to run this project

## 预处理
在 preprocess.ipynb 文件中对GEOM数据集进行预处理

## Intro 
you should run the main_drugs-Copy3_ot-1088.ipynb to get trained model or use our pretraind model to get result.

模型预测的小分子构象保存在 ./confs_save 中，每一个分子为一个文件夹，其中p_*.sdf是我们的模型预测的构象集， t_*.sdf是真实的构象集。

## 可视化
你可以在 show3Dstructure.ipynb 中查看预测的构象 



