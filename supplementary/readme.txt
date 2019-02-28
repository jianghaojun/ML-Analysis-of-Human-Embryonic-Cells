figure文件夹中包含了
（1）三种聚类方法的t-SNE图
（2）不同分类器对E3/E4/E6/E7谱系分类的直方图
（3）kNN对E3/E4/E6/E7谱系分类的t-SNE图

版本信息
python 3.7.1 
scikit-learn 0.20.1

代码使用方法：
在代码中“调参区域”修改选用的聚类算法以及分类算法：
（1）聚类算法：1/2/3分别表示KMeans，高斯混合模型，层次聚类；实验中E5，E7都采用同一种聚类方法，比如method_E5_1，method_E5_2，method_E7_1，method_E7_2都取3(层次聚类)
（2）分类算法：method_LDA表示线性判别分析，依次类推，method_LDA=1则表示采用该方法
注意：数据需要使用在supplementary中提供的embryonic_data_490genes.csv

特征选择参数：
feature_select_method_RFECV =0/1 只有在分类方法为LDA和LinearSVC时起作用

画图：
将参数plot_XXX设置为1则画出相应图，不同变量控制的图已经注释在代码中

