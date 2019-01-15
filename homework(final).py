import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # for plotting
from scipy.stats import mode
import warnings   # to ignore warnings
from sklearn import preprocessing
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error #均方误差
from sklearn.metrics import mean_absolute_error #平方绝对误差
from sklearn.metrics import r2_score#R square
from sklearn.model_selection import  train_test_split
from sklearn.linear_model import LogisticRegression
import re
import seaborn as sns
from xgboost import XGBClassifier
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
data = pd.read_csv('WoodenFloor.csv',encoding='utf-8')
data=data
data=data.fillna(0)
ad=0
for ap in ["店铺评分","商品评价","物流速度","服务态度"]:
    for ad in range(len(data[ap])):
        if data[ap][ad] == "-----":
            data[ap][ad]=0.0
        else:
            data[ap][ad]=float(data[ap][ad])
        ad=ad+1
for ap1 in ["商品毛重"]:
    for ap2 in range(len(data[ap1])):
        data[ap1][ap2]=float(re.sub("\D", "", str(data[ap1][ap2])))
        ap2=ap2+1
for ad2 in ["底板厚度"]:
    for ap3 in range(len(data[ad2])):
        if data[ad2][ap3]=="其他" or data[ad2][ap3]=="NAN":
            data[ad2][ap3] =0.0
        else:
            data[ad2][ap3]=float(re.sub("\D", "", str(data[ad2][ap3])))
        ap2=ap2+1
print(data)
data_hot=data.drop(columns=["晒图","好评","中评","差评","名称","品牌","企口类型","店铺","类别"])
data_hot=pd.DataFrame(data_hot,dtype=float)
corrdf=data_hot.corr()
plt.subplots(figsize=(9, 9)) # 设置画面大小
sns.heatmap(corrdf, annot=True, vmax=1, square=True, cmap="Blues")
plt.show()
print(corrdf)
data_y=data
print(data.info())
#数据预处理

#均值
a1=np.mean(data["晒图"])
a2=np.mean(data["好评"])
a3=np.mean(data["中评"])
a4=np.mean(data["差评"])
print("晒图数据均值",a1)
print("好评数据均值",a2)
print("中评数据均值",a3)
print("差评数据均值",a4)
#中位数
b1=np.median(data["晒图"])
b2=np.median(data["好评"])
b3=np.median(data["中评"])
b4=np.median(data["差评"])
print("晒图数据中位数",b1)
print("好评数据中位数",b2)
print("中评数据中位数",b3)
print("差评数据中位数",b4)
#众数
c1=mode(data["晒图"])
c2=mode(data["好评"])
c3=mode(data["中评"])
c4=mode(data["差评"])
print("晒图数据众数",c1)
print("好评数据众数",c2)
print("中评数据众数",c3)
print("差评数据众数",c4)
#极差
d1=np.ptp(data["晒图"])
d2=np.ptp(data["好评"])
d3=np.ptp(data["中评"])
d4=np.ptp(data["差评"])
print("晒图数据极差",d1)
print("好评数据极差",d2)
print("中评数据极差",d3)
print("差评数据极差",d4)
#方差
e1=np.var(data["晒图"])
e2=np.var(data["好评"])
e3=np.var(data["中评"])
e4=np.var(data["差评"])
print("晒图数据方差",e1)
print("好评数据方差",e2)
print("中评数据方差",e3)
print("差评数据方差",e4)
#频数直方图
for i in ["晒图","好评","中评","差评"]:
    plt.hist(data[i],bins=300,normed=False)
    plt.xlabel(str(i)+"数据(单位：个）")
    plt.ylabel('频数')
    plt.title(str(i)+'频数分布直方图')
    plt.show()
#箱线图
for a in ["晒图","好评","中评","差评"]:
    plt.boxplot(data[a], labels=[a])
    plt.title(str(a)+'箱线图')
    plt.show()
print("数据分析结论：该数据频数分析如上面频数直方图所示，其中差评，好评，晒图，中评数据都集中于0次，由于平均数小于众数，则数据呈现负偏态分布；数据方差，标准差也如上所示，整个数据为离散数据，未呈现连续变化，除此外，在变量相关性上，好评，中评，差评呈现互斥关系，而晒图未发现明确变化，对于其他变量而言未发现明确变化")
#回归模型建立
data=data.drop(columns=["晒图","好评","中评","差评","名称","品牌","企口类型","店铺","类别"])
data_new=data
# data_new.to_csv("data1.csv",encoding="utf-8")
print(data_new.info)
print(data_new.iloc[:,-4:])
data_train=data_new
data_norm=preprocessing.minmax_scale(data_train, feature_range=(0, 1))
data_norm=pd.DataFrame(data_norm)
from sklearn.ensemble import RandomForestClassifier
for label in ["晒图","好评","中评","差评"]:
    train_columns=data_train.columns.values
    X_train, X_test, y_train, y_test = train_test_split(data_norm, data_y[label], train_size=0.8, test_size=0.2)
    # model=RandomForestClassifier(n_estimators=10, max_depth=5,random_state=24,criterion="gini")
    model=XGBClassifier(learning_rate=0.1,n_estimators=100,max_depth=5, min_child_weight = 1,scale_pos_weight=1,colsample_btree=0.8,randosubsample=0.8,m_state=24)
    model.fit(X_train,y_train)
    joblib.dump(model, str(label)+"train_model.m")
    # model=joblib.load( "train_model.m")
    feature_importance = model.feature_importances_
    result1 = np.argsort(feature_importance)
    result=result1[0:10]
    plt.figure(figsize=(64, 64))
    plt.barh(range(len(result1)), feature_importance[result1], align='center')
    plt.yticks(range(len(result1)), train_columns[result1])
    plt.xlabel(str(label)+"变量权重图")
    plt.title('Feature importances')
    plt.draw()
    plt.show()
    print("变量权重如下",feature_importance[result1])
    print(str(label)+"的关键变量如下所示：",train_columns[result1])
    model1 = LogisticRegression(C=1.0,fit_intercept=True,solver='sag',penalty='l2')
    model1.fit(X_train, y_train)
    print(model1.intercept_)
    y_predict = model1.predict(X_test)
    MSE = mean_squared_error(y_test, y_predict)
    RMSE = np.sqrt(MSE)
    R2 = r2_score(y_test, y_predict)
    print("___________________________________________模型指标__________________________________________________________")
    print(str(label)+"的模型RMSE评估值",RMSE)
    print(str(label)+"的模型MSE评估值",MSE)
    print(str(label)+"的模型R2指标值",R2)
print(1)

