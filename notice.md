

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

6 - 引用FiLM方法替代原有门控特征融合，调制公式见图。
Device: cpu
Total samples: 307
  pristine=18, vacancy=213, substitutional=76

Building graphs (vacancy → host structure + site index)...
Graphs built: 307
  vacancy host+site success: 175, fallback: 38
  node dim=8, edge dim=21, cond dim=5

Grouped split (by material id): Train=238, Test=69
  train materials=14, test materials=4, overlap=0
Target standardisation: mean=349.187, std=825.530

>>> Gate: Gated fusion (baseline)
  [gate] Epoch   1 | loss 0.7915 | val MAE 572.02 eV
  [gate] Epoch  10 | loss 0.1162 | val MAE 248.89 eV
  [gate] Epoch  20 | loss 0.0815 | val MAE 165.99 eV
  [gate] Epoch  30 | loss 0.0624 | val MAE 186.64 eV
  [gate] Epoch  40 | loss 0.0676 | val MAE 84.51 eV
  [gate] Epoch  50 | loss 0.0686 | val MAE 105.20 eV
  [gate] Epoch  60 | loss 0.0785 | val MAE 125.24 eV
  [gate] Epoch  70 | loss 0.0476 | val MAE 69.44 eV
  [gate] Epoch  80 | loss 0.0507 | val MAE 81.20 eV
  MAE=66.427  RMSE=106.690  R²=0.9851

>>> FiLM: FiLM modulation
  [film] Epoch   1 | loss 0.7485 | val MAE 502.70 eV
  [film] Epoch  10 | loss 0.1450 | val MAE 185.21 eV
  [film] Epoch  20 | loss 0.1002 | val MAE 163.82 eV
  [film] Epoch  30 | loss 0.0572 | val MAE 77.84 eV
  [film] Epoch  40 | loss 0.0718 | val MAE 101.47 eV
  [film] Epoch  50 | loss 0.0640 | val MAE 170.88 eV
  [film] Epoch  60 | loss 0.0485 | val MAE 87.95 eV
  early stop at epoch 68
  MAE=46.385  RMSE=77.622  R²=0.9921

>>> FiLM_no_charge: FiLM without charge
  [film|noQ] Epoch   1 | loss 0.7207 | val MAE 474.98 eV
  [film|noQ] Epoch  10 | loss 0.2136 | val MAE 134.05 eV
  [film|noQ] Epoch  20 | loss 0.1233 | val MAE 146.61 eV
  [film|noQ] Epoch  30 | loss 0.0749 | val MAE 123.27 eV
  [film|noQ] Epoch  40 | loss 0.0687 | val MAE 95.23 eV
  [film|noQ] Epoch  50 | loss 0.0677 | val MAE 98.21 eV
  [film|noQ] Epoch  60 | loss 0.0676 | val MAE 107.83 eV
  [film|noQ] Epoch  70 | loss 0.0701 | val MAE 90.90 eV
  [film|noQ] Epoch  80 | loss 0.0769 | val MAE 84.93 eV
  MAE=55.151  RMSE=137.994  R²=0.9751

================================================================
Config                   MAE (eV)       RMSE       R²
----------------------------------------------------------------
Gated fusion (baseline)     66.427    106.690   0.9851
FiLM modulation            46.385     77.622   0.9921
FiLM without charge        55.151    137.994   0.9751
================================================================
FiLM_exp finished.