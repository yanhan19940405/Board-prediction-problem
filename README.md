# Board-prediction-problem
本案例使用简易木板商品数据进行标准机器学习流程说明。流程如下：

1.进行数据清洗，将数据中NAN值（空值）在数据量小得时候可以使用0或者均值代替，若数据量大可以直接去掉空值所在输入实例。

2.将非数值型数据处理成可以输入得数值编码。

3.进行特征工程处理阶段，首先画出相应变量热力关联图，确认变量之间得相关性，本例如下所示：
![图1 classreport](https://github.com/yanhan19940405/Board-prediction-problem/blob/master/image/%E7%AC%AC%E4%B8%80%E9%97%AE/%E7%9B%B8%E5%85%B3%E6%80%A7.png)

4.进行人工筛选特征，根据经验知识排除不相干特征项，最后使用随机森林算法或者开方检验进行特征进一步筛选，并画出变量权重分布图：

![图1 classreport](https://github.com/yanhan19940405/Board-prediction-problem/blob/master/image/%E7%AC%AC%E4%BA%8C%E9%97%AE/Figure_.png）
5.根据选定特征进行建模，本例子即确定好评，中评，差评，晒图四种场景下每一种场景不同的特征贡献度，然后建模进行回归预测具体每项指标个数，此处也直接选用随机森林模型进行回归模型构建了并保存模型。

6.其他结果图皆在image文件中。
