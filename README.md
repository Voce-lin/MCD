# MCD
It is a demo of MCD algorithm
# environment
win10
python 3.7
# How to use
1,Download the project , install all the dependencies via requirements.txt
2,Run <font color=red>python train_step.py</font> to complete training, this will update the Cluster.csv and 3D_points.csv
3,RUN <font color=red> python predict_step.py</font> to complete the test step, the detection results of the MCD algorithm on the dataset will be printed out like this：
![image](https://user-images.githubusercontent.com/42335842/120985431-20c2d680-c76b-11eb-8c60-24b4252a3fcd.png)
# Notice
The default dataset is the robin dataset, if you want to test other datasets, copy the dataset to the "data/" path and replace the file path on line 19 in "train_step.py". For example, modify it to:
<font color=red> filepath=f'data/kpi_dataset/'</font>
The csv file format of the dataset needs to be the same as the robin dataset.
