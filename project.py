# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 19:06:21 2019

@author: hjjiang
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from pyclustering.cluster.kmedoids import kmedoids
from sklearn.preprocessing import normalize
from sklearn.mixture import GaussianMixture 
from sklearn.feature_selection import RFECV
from sklearn.metrics import accuracy_score

##############################################################################
#------------------调参区域-------------------
#无监督学习
plot_E5_1 = 0 #E3-E7的TSNE图
plot_E5_2 = 0 #E5的ICM,TE的TSNE图
plot_E5_3 = 0 #E5的EPI,PE的TSNE图
method_E5_1 = 3 #分离E5的TE,ICM的聚类方法 见下方聚类方法
method_E5_2 = 3 #分离E5的EPI,PE的聚类方法 见下方聚类方法

plot_E7_1 = 0 #E7的ICM,TE的TSNE图
plot_E7_2 = 0 #E7的EPI,PE的TSNE图
method_E7_1 = 3 #分离E7的ICM,TE的聚类方法 见下方聚类方法
method_E7_2 = 3 #分离E7的EPI,PE的聚类方法 见下方聚类方法
'''
聚类方法 
1:KMeans
2:GMM
3:AgglomerativeClustering
'''
#监督学习
method_LDA = 0
method_LinearSVC = 1
method_SVM = 0
method_RandomForest = 0
method_DecisionTree = 0
method_MultilayerPerceptron = 0
method_NearestNeighbor = 0

#画出E3,E4,E6,E7谱系分类情况 直方图
plot_LDA = 0
plot_LinearSVC = 1
plot_SVM = 0
plot_RandomForest = 0
plot_DecisionTree = 0
plot_MultilayerPerceptron = 0
plot_NearestNeighbor = 0

#特征选择 只有LDA LinearSVC可以设置feature_select_method_RFECV=1
feature_select_method_RFECV = 0 #recursive feature elimination
##############################################################################
#------数据读取，用我给的csv，已经把label放进去了-------
dat = pd.read_csv("./embryonic_data_490genes.csv", index_col = 0)
allX, ally = dat.iloc[:,1:], dat['label'] #提取数据和标签

#--------数据预处理---------------------------
allX = np.log10(allX + 1) #对数据取log

##############################################################################
#------------------无监督学习-------------------
#-----------TSNE，目的是比较每天的差异---------------------
X_2d = TSNE(n_components=2,random_state=0).fit_transform(allX)
if plot_E5_1:
    plt.figure(figsize=(6, 5))
    colors = 'r', 'g', 'b', 'c', 'm'
    la=np.array([3,4,5,6,7])
    for i, c, label in zip(la, colors, la):
        plt.scatter(X_2d[ally == i, 0], X_2d[ally == i, 1], c=c, label=label)
    plt.legend()
    plt.show()

#------E5分离pre-lineage-----------------------------------
threshold = 15 #根据上面的图人为设定的pre-lineage的界限
    
E5_without_pre = allX[(X_2d[:,0] < threshold) & (ally == 5)]  #E5_without_pre
allX['lineage'] = 0 #标签初始化为0   TE->1;ICM ->2(进一步分裂):EPI->2, PE->3; pre-lineage->4
allX['lineage'][(X_2d[:,0] > threshold) & (ally == 5)] = 4 # 给E5的pre-lineage贴标签

pca = PCA(n_components=3) #特征提取
E5_without_pre_processed = pca.fit_transform(E5_without_pre)

#------聚类：E5_without_pre，分成2类ICM和TE----------------
if method_E5_1 == 1:
    estimator = KMeans(n_clusters=2).fit(E5_without_pre_processed)
    label_pred = estimator.labels_  # 获取聚类标签 根据论文，多的是TE，少的是ICM(EPI+PE)
    
if method_E5_1 == 2:
    estimator = GaussianMixture(n_components=2, covariance_type='diag', random_state=0).fit(E5_without_pre_processed)
    label_pred = estimator.predict(E5_without_pre_processed)
    
if method_E5_1 == 3: 
    estimator=AgglomerativeClustering(affinity='euclidean', compute_full_tree='auto',
            connectivity=None, linkage='ward', memory=None, n_clusters=2).fit(E5_without_pre_processed)
    label_pred = estimator.labels_ 

ind = int(sum(label_pred==0) < sum(label_pred==1)) #指示因子
allX['lineage'][(X_2d[:,0] < threshold) & (ally == 5)] = label_pred*(1-2*ind) +ind + 1

'''
这里有一个逻辑判断,由于少的那一类的标签，要定义为ICM，即贴上“2”的标签
1.label_pred==0 说明0少一些，0表示ICM
    ind = 1
    label_pred(1-2ind)+ind+1 = -label_pred+2   0(ICM)->2, 1(TE)->1
2.label_pred==1 说明1少一些，1表示ICM
    ind = 0
    label_pred(1-2ind)+ind+1 = label_pred+1    1(ICM)->2, 0(TE)->1       
TE: 1  ICM: 2
'''

if plot_E5_2:
    X_2d = TSNE(n_components=2,random_state=0).fit_transform(E5_without_pre)
    plt.figure(figsize=(6, 5))
    colors = 'r', 'g'
    la=np.array([0,1])
    for i, c, label in zip(la, colors, la):
        plt.scatter(X_2d[label_pred == i, 0], X_2d[label_pred == i, 1], c=c, label=label)
    plt.legend()
    plt.title('ICM-TE t-SNE plot')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.show()

#------聚类：E5_ICM，分成2类EPI和PE----------------
E5_ICM = allX[allX['lineage']== 2]  #选出E5中所有ICM
E5_ICM = E5_ICM.iloc[:,:-1]

if method_E5_2 == 1: #KNN
    estimator = KMeans(n_clusters=2).fit(E5_ICM)  #获取聚类标签 根据论文，多的是EPI，少的是PE
    label_pred_EPI_PE = estimator.labels_  # 获取聚类标签 根据论文，多的是TE，少的是ICM(EPI+PE)
    
if method_E5_2 == 2: #GMM
    estimator = GaussianMixture(n_components=2, covariance_type='diag', random_state=0).fit(E5_ICM)
    label_pred_EPI_PE = estimator.predict(E5_ICM)   
    
if method_E5_2 == 3: #AgglomerativeClustering
    estimator=AgglomerativeClustering(affinity='euclidean', compute_full_tree='auto',
            connectivity=None, linkage='ward', memory=None, n_clusters=2).fit(E5_ICM)
    label_pred_EPI_PE = estimator.labels_ 
    
    
ind = int(sum(label_pred_EPI_PE==0) < sum(label_pred_EPI_PE==1)) #指示因子
allX['lineage'][allX['lineage']== 2] = label_pred_EPI_PE*(1-2*ind) +ind + 2 #重新贴标签，EPI=2,PE=3

if plot_E5_3:
    X_2d_EPI_PE = TSNE(n_components=2,random_state=0).fit_transform(E5_ICM)
    plt.figure(figsize=(6, 5))
    colors = 'r', 'g'
    la=np.array([0,1])
    for i, c, label in zip(la, colors, la):
        plt.scatter(X_2d_EPI_PE[label_pred_EPI_PE == i, 0], X_2d_EPI_PE[label_pred_EPI_PE == i, 1], c=c, label=label)
    plt.legend()
    plt.title('EPI-PE t-SNE plot')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.show()

#E5结果统计
print('E5 TE NUM:', sum(allX['lineage']==1))
print('E5 EPI NUM:', sum(allX['lineage']==2))
print('E5 PE NUM:', sum(allX['lineage']==3))
print('E5 PRE NUM:', sum(allX['lineage']==4), '\n')

#E5聚类结果提取
new5= allX[allX['lineage'] != 0]
new_data5, new_label5 = new5.iloc[:,:490], new5['lineage']

##############################################################################
#----------------------------E7聚类-----------------------------------
allX['lineage'] = 0 #标签初始化为0   TE->1;ICM ->2(进一步分裂):EPI->2, PE->3;
E7_data = allX[ally == 7]
E7_data = E7_data.iloc[:,:-1]
E7_data = normalize(E7_data)
pca = PCA(n_components=3) #特征提取
E7_data = pca.fit_transform(E7_data)

#------聚类:分成2类ICM和TE----------------
if method_E7_1 == 1:
    estimator = KMeans(n_clusters=2).fit(E7_data)
    label_pred = estimator.labels_  # 获取聚类标签 根据论文，多的是TE，少的是ICM(EPI+PE)
    
if method_E7_1 == 2:
    estimator = GaussianMixture(n_components=2, covariance_type='diag', random_state=0).fit(E7_data)
    label_pred = estimator.predict(E7_data)
    
if method_E7_1 == 3: 
    estimator = AgglomerativeClustering(affinity='euclidean', compute_full_tree='auto',
            connectivity=None, linkage='ward', memory=None, n_clusters=2).fit(E7_data)
    label_pred = estimator.labels_

ind = int(sum(label_pred == 0) < sum(label_pred == 1)) #指示因子
allX['lineage'][ally == 7] = label_pred*(1-2*ind) +ind + 1

if plot_E7_1:
    X_2d = TSNE(n_components=2,random_state=0).fit_transform(allX[ally == 7].iloc[:,:-1])
    plt.figure(figsize=(6, 5))
    colors = 'r', 'g'
    la=np.array([0,1])
    for i, c, label in zip(la, colors, la):
        plt.scatter(X_2d[label_pred == i, 0], X_2d[label_pred == i, 1], c=c, label=label)
    plt.legend()
    plt.title('ICM-TE t-SNE plot')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.show()

#------聚类：E7_ICM，分成2类EPI和PE----------------
E7_ICM = allX[allX['lineage']== 2]  #选出E7中所有ICM
E7_ICM = E7_ICM.iloc[:,:-1]

if method_E7_2 == 1: #KNN
    estimator = KMeans(n_clusters=2).fit(E7_ICM)  #获取聚类标签 根据论文，多的是EPI，少的是PE
    label_pred_EPI_PE = estimator.labels_  # 获取聚类标签 根据论文，多的是TE，少的是ICM(EPI+PE)
    
if method_E7_2 == 2: #GMM
    estimator = GaussianMixture(n_components=2, covariance_type='diag', random_state=0).fit(E7_ICM)
    label_pred_EPI_PE = estimator.predict(E7_ICM)   
    
if method_E7_2 == 3: #AgglomerativeClustering
    estimator = AgglomerativeClustering(affinity='euclidean', compute_full_tree='auto',
            connectivity=None, linkage='ward', memory=None, n_clusters=2).fit(E7_ICM)
    label_pred_EPI_PE = estimator.labels_ 
    
ind = int(sum(label_pred_EPI_PE==0) < sum(label_pred_EPI_PE==1)) #指示因子
allX['lineage'][allX['lineage']== 2] = label_pred_EPI_PE*(1-2*ind) +ind + 2 #重新贴标签，EPI=2,PE=3

if plot_E7_2:
    X_2d_EPI_PE = TSNE(n_components=2,random_state=0).fit_transform(E7_ICM)
    plt.figure(figsize=(6, 5))
    colors = 'r', 'g'
    la=np.array([0,1])
    for i, c, label in zip(la, colors, la):
        plt.scatter(X_2d_EPI_PE[label_pred_EPI_PE == i, 0], X_2d_EPI_PE[label_pred_EPI_PE == i, 1], c=c, label=label)
    plt.legend()
    plt.title('EPI-PE t-SNE plot')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.show()

#E7结果统计
print('E7 TE NUM:', sum(allX['lineage']==1))
print('E7 EPI NUM:', sum(allX['lineage']==2))
print('E7 PE NUM:', sum(allX['lineage']==3))

#E7聚类结果提取
new7= allX[allX['lineage'] != 0]
new_data7, new_label7 = new7.iloc[:,:490], new7['lineage']

##############################################################################
#------------------监督学习-------------------
dat_3 = allX[ally == 3].iloc[:,:-1]
dat_4 = allX[ally == 4].iloc[:,:-1]
dat_6 = allX[ally == 6].iloc[:,:-1]
dat_7 = allX[ally == 7].iloc[:,:-1]

#Linear Discriminant Analysis with feature selection
if method_LDA:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    if feature_select_method_RFECV:
        LDA = LinearDiscriminantAnalysis()
        rfe = RFECV(estimator=LDA, step=10, cv=8)
        rfe = rfe.fit(new_data5, new_label5)
        index = np.where(rfe.support_ == True)[0]
        new_data5_selected = new_data5.iloc[:,index]
        LDA.fit(new_data5_selected, new_label5)
        LDA_prediction3=LDA.predict(dat_3.iloc[:,index])
        LDA_prediction4=LDA.predict(dat_4.iloc[:,index])
        LDA_prediction6=LDA.predict(dat_6.iloc[:,index])
        LDA_prediction7=LDA.predict(dat_7.iloc[:,index])
        print('----------------------RFECV----------------------')
    else:   
        LDA = LinearDiscriminantAnalysis()
        new_data5_selected = new_data5
        LDA.fit(new_data5_selected, new_label5)
        LDA_prediction3=LDA.predict(dat_3)
        LDA_prediction4=LDA.predict(dat_4)
        LDA_prediction6=LDA.predict(dat_6)
        LDA_prediction7=LDA.predict(dat_7)
    
    if plot_LDA:
        for i in range(4):
            plt.figure(figsize=(6, 5))
            name_list = ['TE','EPI','PE','PRE']
            if i < 2:
                num_list = [sum(eval('LDA_prediction' + str(i+3)) == 1),sum(eval('LDA_prediction' + str(i+3)) == 2),sum(eval('LDA_prediction' + str(i+3)) == 3),sum(eval('LDA_prediction' + str(i+3)) == 4)]
            else:
                num_list = [sum(eval('LDA_prediction' + str(i+4)) == 1),sum(eval('LDA_prediction' + str(i+4)) == 2),sum(eval('LDA_prediction' + str(i+4)) == 3),sum(eval('LDA_prediction' + str(i+4)) == 4)]
            plt.bar(range(len(num_list)), num_list,color='rgbc',tick_label=name_list)
            plt.title('LDA-'+'E'+str(i+3)) if i <2 else plt.title('LDA-'+'E'+str(i+4))
            for a,b in zip(range(len(num_list)), num_list):
                plt.text(a, b+0.05, '%.0f' % b, ha='center', va='bottom', fontsize=10)
            plt.show()

    scores_LDA = cross_val_score(LDA, new_data5_selected, new_label5, cv=8)
    scores_LDA_mean=scores_LDA.mean()
    print('---------scores_LDA_mean---------:', scores_LDA_mean)
    scores_LDA_E7_test = accuracy_score(new_label7, LDA_prediction7)
    print('---------scores_LDA_E7_test---------:', scores_LDA_E7_test)

#SVM
if method_SVM:
    from sklearn.svm import SVC
    #if feature_select_method_REFCV: svm does not expose "coef_" or "feature_importances_" attributes
    SVM = SVC(kernel='rbf', gamma='auto')
    SVM.fit(new_data5, new_label5)
    SVM_prediction3=SVM.predict(dat_3)
    SVM_prediction4=SVM.predict(dat_4)
    SVM_prediction6=SVM.predict(dat_6)
    SVM_prediction7=SVM.predict(dat_7)
    scores_SVM = cross_val_score(SVM, new_data5, new_label5, cv=8)
    scores_SVM_mean=scores_SVM.mean()
    print('---------scores_SVM_mean---------:', scores_SVM_mean)
    scores_SVM_E7_test = accuracy_score(new_label7, SVM_prediction7)
    print('---------scores_SVM_E7_test---------:', scores_SVM_E7_test)
    
    if plot_SVM:
        for i in range(4):
            plt.figure(figsize=(6, 5))
            name_list = ['TE','EPI','PE','PRE']
            if i < 2:
                num_list = [sum(eval('SVM_prediction' + str(i+3)) == 1),sum(eval('SVM_prediction' + str(i+3)) == 2),sum(eval('SVM_prediction' + str(i+3)) == 3),sum(eval('SVM_prediction' + str(i+3)) == 4)]
            else:
                num_list = [sum(eval('SVM_prediction' + str(i+4)) == 1),sum(eval('SVM_prediction' + str(i+4)) == 2),sum(eval('SVM_prediction' + str(i+4)) == 3),sum(eval('SVM_prediction' + str(i+4)) == 4)]
            plt.bar(range(len(num_list)), num_list,color='rgbc',tick_label=name_list)
            plt.title('SVM-'+'E'+str(i+3)) if i <2 else plt.title('SVM-'+'E'+str(i+4))
            for a,b in zip(range(len(num_list)), num_list):
                plt.text(a, b+0.05, '%.0f' % b, ha='center', va='bottom', fontsize=10)
            plt.show()

#LinearSVC
if method_LinearSVC:
    from sklearn.svm import LinearSVC
    if feature_select_method_RFECV:
        LSVC = LinearSVC(C=0.01, penalty='l1', dual=False, max_iter=5000)
        rfe = RFECV(estimator=LSVC, step=100, cv=8)
        rfe = rfe.fit(new_data5, new_label5)
        index = np.where(rfe.support_ == True)[0]
        new_data5_selected = new_data5.iloc[:,index]
        LSVC.fit(new_data5_selected, new_label5)
        LSVC_prediction3=LSVC.predict(dat_3.iloc[:,index])
        LSVC_prediction4=LSVC.predict(dat_4.iloc[:,index])
        LSVC_prediction6=LSVC.predict(dat_6.iloc[:,index])
        LSVC_prediction7=LSVC.predict(dat_7.iloc[:,index])
        print('----------------------RFECV----------------------')
    
    else:
        LSVC = LinearSVC(C=0.01, penalty='l1', dual=False, max_iter=5000)
        new_data5_selected = new_data5
        LSVC.fit(new_data5_selected, new_label5)
        LSVC_prediction3=LSVC.predict(dat_3)
        LSVC_prediction4=LSVC.predict(dat_4)
        LSVC_prediction6=LSVC.predict(dat_6)
        LSVC_prediction7=LSVC.predict(dat_7)
    
    if plot_LinearSVC:
        for i in range(4):
            plt.figure(figsize=(6, 5))
            name_list = ['TE','EPI','PE','PRE']
            if i < 2:
                num_list = [sum(eval('LSVC_prediction' + str(i+3)) == 1),sum(eval('LSVC_prediction' + str(i+3)) == 2),sum(eval('LSVC_prediction' + str(i+3)) == 3),sum(eval('LSVC_prediction' + str(i+3)) == 4)]
            else:
                num_list = [sum(eval('LSVC_prediction' + str(i+4)) == 1),sum(eval('LSVC_prediction' + str(i+4)) == 2),sum(eval('LSVC_prediction' + str(i+4)) == 3),sum(eval('LSVC_prediction' + str(i+4)) == 4)]
            plt.bar(range(len(num_list)), num_list,color='rgbc',tick_label=name_list)
            plt.title('LinearSVC-'+'E'+str(i+3)) if i <2 else plt.title('LinearSVC-'+'E'+str(i+4))
            for a,b in zip(range(len(num_list)), num_list):
                plt.text(a, b+0.05, '%.0f' % b, ha='center', va='bottom', fontsize=10)
            plt.show()
            
    scores_LSVC = cross_val_score(LSVC, new_data5_selected, new_label5, cv=8)
    scores_LSVC_mean=scores_LSVC.mean()
    print('---------scores_LSVC_mean---------:', scores_LSVC_mean)
    scores_LSVC_E7_test = accuracy_score(new_label7, LSVC_prediction7)
    print('---------scores_LSVC_E7_test---------:', scores_LSVC_E7_test)


#Random Forest
if method_RandomForest:
    from sklearn.ensemble import RandomForestClassifier
    RF=RandomForestClassifier(n_estimators=100, max_depth=10,random_state=0)
    scores_RF = cross_val_score(RF, new_data5, new_label5, cv=8)
    scores_RF_mean=scores_RF.mean()
    print('---------scores_RF_mean---------:', scores_RF_mean)
    RF.fit(new_data5,new_label5)
    RF_prediction3=RF.predict(dat_3)
    RF_prediction4=RF.predict(dat_4)
    RF_prediction6=RF.predict(dat_6)
    RF_prediction7=RF.predict(dat_7)
    scores_RF_E7_test = accuracy_score(new_label7, RF_prediction7)
    print('---------scores_RF_E7_test---------:', scores_RF_E7_test)
    
    if plot_RandomForest:
        for i in range(4):
            plt.figure(figsize=(6, 5))
            name_list = ['TE','EPI','PE','PRE']
            if i < 2:
                num_list = [sum(eval('RF_prediction' + str(i+3)) == 1),sum(eval('RF_prediction' + str(i+3)) == 2),sum(eval('RF_prediction' + str(i+3)) == 3),sum(eval('RF_prediction' + str(i+3)) == 4)]
            else:
                num_list = [sum(eval('RF_prediction' + str(i+4)) == 1),sum(eval('RF_prediction' + str(i+4)) == 2),sum(eval('RF_prediction' + str(i+4)) == 3),sum(eval('RF_prediction' + str(i+4)) == 4)]
            plt.bar(range(len(num_list)), num_list,color='rgbc',tick_label=name_list)
            plt.title('RandomForest-'+'E'+str(i+3)) if i <2 else plt.title('RandomForest-'+'E'+str(i+4))
            for a,b in zip(range(len(num_list)), num_list):
                plt.text(a, b+0.05, '%.0f' % b, ha='center', va='bottom', fontsize=10)
            plt.show()

#Decision Tree
if method_DecisionTree:
    from sklearn import tree
    DT= tree.DecisionTreeClassifier()
    scores_DT = cross_val_score(DT, new_data5, new_label5, cv=8)
    scores_DT_mean=scores_DT.mean()
    print('---------scores_DT_mean---------:', scores_DT_mean)
    DT.fit(new_data5,new_label5)
    DT_prediction3=DT.predict(dat_3)
    DT_prediction4=DT.predict(dat_4)
    DT_prediction6=DT.predict(dat_6)
    DT_prediction7=DT.predict(dat_7)
    scores_DT_E7_test = accuracy_score(new_label7, DT_prediction7)
    print('---------scores_DT_E7_test---------:', scores_DT_E7_test)
    
    if plot_DecisionTree:
        for i in range(4):
            plt.figure(figsize=(6, 5))
            name_list = ['TE','EPI','PE','PRE']
            if i < 2:
                num_list = [sum(eval('DT_prediction' + str(i+3)) == 1),sum(eval('DT_prediction' + str(i+3)) == 2),sum(eval('DT_prediction' + str(i+3)) == 3),sum(eval('DT_prediction' + str(i+3)) == 4)]
            else:
                num_list = [sum(eval('DT_prediction' + str(i+4)) == 1),sum(eval('DT_prediction' + str(i+4)) == 2),sum(eval('DT_prediction' + str(i+4)) == 3),sum(eval('DT_prediction' + str(i+4)) == 4)]
            plt.bar(range(len(num_list)), num_list,color='rgbc',tick_label=name_list)
            plt.title('DecisionTree-'+'E'+str(i+3)) if i <2 else plt.title('DecisionTree-'+'E'+str(i+4))
            for a,b in zip(range(len(num_list)), num_list):
                plt.text(a, b+0.05, '%.0f' % b, ha='center', va='bottom', fontsize=10)
            plt.show()

#MLP
if method_MultilayerPerceptron:
    from sklearn.neural_network import MLPClassifier
    MLP= MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=9, random_state=1)
    scores_MLP = cross_val_score(MLP, new_data5, new_label5, cv=8)
    scores_MLP_mean=scores_MLP.mean()
    print('---------scores_MLP_mean---------:', scores_MLP_mean)
    MLP.fit(new_data5,new_label5)
    MLP_prediction3=MLP.predict(dat_3)
    MLP_prediction4=MLP.predict(dat_4)
    MLP_prediction6=MLP.predict(dat_6)
    MLP_prediction7=MLP.predict(dat_7)
    scores_MLP_E7_test = accuracy_score(new_label7, MLP_prediction7)
    print('---------scores_MLP_E7_test---------:', scores_MLP_E7_test)
    
    if plot_MultilayerPerceptron:
        for i in range(4):
            plt.figure(figsize=(6, 5))
            name_list = ['TE','EPI','PE','PRE']
            if i < 2:
                num_list = [sum(eval('MLP_prediction' + str(i+3)) == 1),sum(eval('MLP_prediction' + str(i+3)) == 2),sum(eval('MLP_prediction' + str(i+3)) == 3),sum(eval('MLP_prediction' + str(i+3)) == 4)]
            else:
                num_list = [sum(eval('MLP_prediction' + str(i+4)) == 1),sum(eval('MLP_prediction' + str(i+4)) == 2),sum(eval('MLP_prediction' + str(i+4)) == 3),sum(eval('MLP_prediction' + str(i+4)) == 4)]
            plt.bar(range(len(num_list)), num_list,color='rgbc',tick_label=name_list)
            plt.title('MultilayerPerceptron-'+'E'+str(i+3)) if i <2 else plt.title('MultilayerPerceptron-'+'E'+str(i+4))
            for a,b in zip(range(len(num_list)), num_list):
                plt.text(a, b+0.05, '%.0f' % b, ha='center', va='bottom', fontsize=10)
            plt.show()

#Nearest Neighbour Classification
if method_NearestNeighbor:
    from sklearn.neighbors import KNeighborsClassifier
    KNN = KNeighborsClassifier(n_neighbors=6)
    scores_KNN = cross_val_score(KNN, new_data5, new_label5, cv=8)
    scores_KNN_mean=scores_KNN.mean()
    print('---------scores_KNN_mean---------:', scores_KNN_mean)
    KNN.fit(new_data5,new_label5)
    KNN_prediction3=KNN.predict(dat_3)
    KNN_prediction4=KNN.predict(dat_4)
    KNN_prediction6=KNN.predict(dat_6)
    KNN_prediction7=KNN.predict(dat_7)
    scores_KNN_E7_test = accuracy_score(new_label7, KNN_prediction7)
    print('---------scores_KNN_E7_test---------:', scores_KNN_E7_test)
    
    if plot_NearestNeighbor:
        for i in range(4):
            plt.figure(figsize=(6, 5))
            name_list = ['TE','EPI','PE','PRE']
            if i < 2:
                num_list = [sum(eval('KNN_prediction' + str(i+3)) == 1),sum(eval('KNN_prediction' + str(i+3)) == 2),sum(eval('KNN_prediction' + str(i+3)) == 3),sum(eval('KNN_prediction' + str(i+3)) == 4)]
            else:
                num_list = [sum(eval('KNN_prediction' + str(i+4)) == 1),sum(eval('KNN_prediction' + str(i+4)) == 2),sum(eval('KNN_prediction' + str(i+4)) == 3),sum(eval('KNN_prediction' + str(i+4)) == 4)]
            plt.bar(range(len(num_list)), num_list,color='rgbc',tick_label=name_list)
            plt.title('NearestNeighbor-'+'E'+str(i+3)) if i <2 else plt.title('NearestNeighbor-'+'E'+str(i+4))
            for a,b in zip(range(len(num_list)), num_list):
                plt.text(a, b+0.05, '%.0f' % b, ha='center', va='bottom', fontsize=10)
            plt.show()

