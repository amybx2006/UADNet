<!--
 * @Author: 丁铖,白雪琼
 * @Date: 2026-04-20 16:29:26
 * @Description: 
 * 
 * Copyright (c) 2026 by 丁铖,白雪琼, All Rights Reserved. 
-->
# UAD-Net文件说明
## 数据集
使用CAMO，COD10K，NC4K三个数据集进行训练和测试，相关设置如下：
| Datasets  | CAMO        | COD10K      | NC4K          | sum |
| :---      | :----:      |    :----:   |          ---: |---: |
| train     | 1000        | 3040        |  0            |4040 |
| test      | 250         | 2026        | 4121          |6397 |
| sum       | 1250        | 5056        | 4121          |10137|

在训练及测试前，将数据集进行处理如下： 
数据文件夹结构如下：  
![alt text](image-1.png)  
其中，所有训练集图片全部整合进TrainDataset-Imgs文件夹，GT为真值图，EDGE为伪装目标边缘图像；EDGE，GT中图像名称与Imgs中文件名称一一对应，但文件后缀不同，训练及测试中对文件对应的相关处理可参考utils--dataloader.py、dataloader_edge.py。

## 依赖包
见 requirements.txt

## 运行程序

1. 环境配置:
    
    + 创建Pytorch环境：具体过程略
    
    + 安装依赖包: 程序依赖包见`requirements.txt`.

2. 下载数据集:

    + 测试集地址： [download link (Google Drive)](https://drive.google.com/file/d/1SLRB5Wg1Hdy7CQ74s3mTQ3ChhjFRSFdZ/view?usp=sharing).
    
    + 训练集地址： [download link (Google Drive)](https://drive.google.com/file/d/1Kifp7I0n9dlWKXXNIbN7kgyokoRY4Yz7/view?usp=sharing).
    
    + 模型参数，可用于推理，评估。位于： `./checkpoints\Net_epoch_best_22.pth`, 
    

3. 训练设置:

    + 请修改`MyTrain.py`中的相关参数， `--train_save` and `--train_path` in `MyTrain.py`.
  

4. Testing Configuration:

    + After you download all the pre-trained model and testing dataset, just run `MyTest.py` to generate the final prediction map: 
    replace your trained model directory (`--pth_path`).

## 评估模型
1. 评估步骤
   + 首先通过测试代码，生成待评估的图像
   + 修改评估程序中相关参数，并运行评估程序  
2. 评估程序地址
   + 可以选择使用MATLAB代码进行评估(revised from [link](https://github.com/DengPingFan/CODToolbox)), .
   + 如使用python进行评估，需要安装相应库[link](https://github.com/lartpang/PySODMetrics) by `pip install pysodmetrics`.
   作者根据需要，也以相关库为基础，修改了Python代码，见 `./metrics/eval.py` 运行后自动将评估结果输出到xlx文件中
