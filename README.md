<!--
 * @Author: Ding Cheng, Bai Xueqiong
 * @Date: 2026-04-20 16:29:26
 * @Description: 
 * 
 * Copyright (c) 2026 by Ding Cheng, Bai Xueqiong, All Rights Reserved. 
-->

# UAD-Net File Description

## Datasets

Three datasets — CAMO, COD10K, and NC4K — are used for training and testing. The detailed configuration is as follows:

| Datasets  | CAMO        | COD10K      | NC4K          | Sum  |
| :---      | :----:      |    :----:   |          ---: |---:  |
| Train     | 1000        | 3040        | 0             | 4040 |
| Test      | 250         | 2026        | 4121          | 6397 |
| Sum       | 1250        | 5056        | 4121          | 10137|

MC1K Dataset will be made available on request. <img width="432" height="23" alt="image" src="https://github.com/user-attachments/assets/dd7f9f73-5074-486d-99a2-21c5f99e845d" />

Before training and testing, the datasets are organized as follows:

The data folder structure is shown below:  
![alt text](image-1.png)

All training images are consolidated into the `TrainDataset-Imgs` folder. `GT` contains ground truth maps, and `EDGE` contains edge maps of camouflaged objects. The filenames in `EDGE` and `GT` correspond one-to-one with those in `Imgs`, but with different file extensions. For details on how file correspondence is handled during training and testing, please refer to `utils/dataloader.py` and `dataloader_edge.py`.

## Dependencies

See `requirements.txt`.

## Running the Program

1. **Environment Setup:**

   - Create a PyTorch environment *(details omitted)*.
   - Install dependencies: `pip install -r requirements.txt`.

2. **Download Datasets:**

   - Test set: [Download link (Google Drive)](https://drive.google.com/file/d/1SLRB5Wg1Hdy7CQ74s3mTQ3ChhjFRSFdZ/view?usp=sharing)
   - Train set: [Download link (Google Drive)](https://drive.google.com/file/d/1Kifp7I0n9dlWKXXNIbN7kgyokoRY4Yz7/view?usp=sharing)
   - Pre-trained model weights (for inference and evaluation): `./checkpoints/Net_epoch_best_22.pth`

3. **Training Configuration:**

   - Modify the relevant parameters in `MyTrain.py`, including `--train_save` and `--train_path`.

4. **Testing Configuration:**

   - After downloading all pre-trained model weights and the testing dataset, run `MyTest.py` to generate the final prediction maps. Replace the trained model directory with `--pth_path`.

## Model Evaluation

1. **Evaluation Steps:**
   - First, run the testing script to generate the prediction images to be evaluated.
   - Modify the relevant parameters in the evaluation script and run it.

2. **Evaluation Tools:**
   - You may use the **MATLAB** evaluation code (revised from [this repository](https://github.com/DengPingFan/CODToolbox)).
   - For **Python**-based evaluation, install the required library via:
     ```
     pip install pysodmetrics
     ```
     Reference: [PySODMetrics](https://github.com/lartpang/PySODMetrics)
   - The authors have also modified the Python evaluation code based on this library. See `./metrics/eval.py`. Upon execution, the evaluation results are automatically exported to an `.xlsx` file.
