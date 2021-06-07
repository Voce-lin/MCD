# MCD
It is a demo of MCD algorithm
# environment
win10<br>
python 3.7
# How to use
1,Download the project , install all the dependencies via requirements.txt<br><br>
2,Run ```python train_step.py ``` in CMD to complete the training, this will update the Cluster.csv and 3D_points.csv<br><br>
3,RUN ```python predict_step.py ``` in CMD to complete the test step, the detection results of the MCD algorithm on the dataset will be printed out like this：<br>
![image](https://user-images.githubusercontent.com/42335842/120985431-20c2d680-c76b-11eb-8c60-24b4252a3fcd.png)
# Notice
The default dataset is the robin dataset, if you want to test other datasets, copy the dataset to the "data/" path and replace the file path on line 19 in "train_step.py". For example, modify it to:<br>
``` filepath=f'data/kpi_dataset/' ```<br>
The csv file format of the dataset needs to be the same as the robin dataset.
# algorithm
If you want to see the details of the algorithm, all the operators are in the utils.py file, they are all very simple。
