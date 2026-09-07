

实验记录：
1 - 以soap机器学习方法为基线，进行了比对实验，加入type one hot特征，r2为0.99 
  
2 - 进行了消融实验 去除type one hot特征后r2暴跌，图特征贡献小，模型依赖性过强

3 - 去除type one hot特征进行重训，修改此项为分类任务目标，同时增加epoch至100；早停于90轮且40轮之后无明显提升。表现相对不理想：
    Energy  MAE  : 134.541 eV
    Energy  RMSE : 258.401 eV
    Energy  R²   : 0.9043
    Type    Acc  : 75.8%

4 - 加入了mulliken电荷特征作为辅助输入。

5 - 弱化坐标信息并除去type one-hot特征，强迫模型进行纯粹图学习。第二次特征消融实验。