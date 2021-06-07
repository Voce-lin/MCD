import os,multiprocessing
import numpy as np
import pandas as pd
from utils import utils
from sklearn import preprocessing

def load_data(path):
    dataframe=pd.read_csv(path).dropna(how='any')
    value=np.array(dataframe['value'])
    value=value-min(value)/(max(value)-min(value))
    try:
        label=np.array(dataframe['label'])
    except:
        label = np.array(dataframe['is_anomaly'])
    return value,label

def get_all_filepath():
    all_path=[]
    filepath=f'data/robin_dataset/'
    fileList = os.listdir(filepath)
    for file in fileList:
        path=filepath+file
        all_path.append(path)
    return all_path

def cal_xyz(feature,util):
    item=feature
    unif_ts_1 = (item - np.min(item)) / (np.max(item) - np.min(item))
    if np.max(item)==np.min(item):
        unif_ts_1=np.ones(len(item))
    x = util.cal_wavelet(unif_ts_1)
    y = np.std(unif_ts_1)
    if len(unif_ts_1)>4000:
        util=utils()
        unif_ts_1=util.down_sampling_to_len(unif_ts_1,length=4000)
    z = util.cal_T(unif_ts_1)
    return x,y,z

if __name__=="__main__":
    util=utils()
    pool = multiprocessing.Pool(processes=3)
    all_path=get_all_filepath()
    print(all_path)
    all_features=[load_data(path)[0] for path in all_path]
    #all_features_=[util.down_sampling_to_len(feature) for feature in all_features]
    # all_features_=[feature for feature in all_features]
    # print('load done')
    # cls = KMedoids(n_clusters=3,metric='wavelet')
    # pred = cls.fit_predict(all_features_, randomstate=7)
    # print(pred)

    # 计算投影坐标点
    all_points,all_x,all_y,all_z,all_progress=[],[],[],[],[]
    for i in range(len(all_features)):
        all_progress.append(pool.apply_async(cal_xyz, (all_features[i],util)))
    i=0
    for item in all_progress:
        i += 1
        x,y,z=item.get()
        all_points.append([15 + np.arctan(x - 14),y,z])
        all_x.append(15 + np.arctan(x - 14))
        all_y.append(y)
        all_z.append(z)
        print(i, end=' ')
    max_x, max_y, max_z = max(all_x), max(all_y), max(all_z)
    min_x, min_y, min_z = min(all_x), min(all_y), min(all_z)
    for point in all_points:
        point[0]=100*(point[0]-min_x)/(max_x-min_x)
        point[1]=100*(point[1]-min_y)/(max_y-min_y)
        point[2]=100*(point[2]-min_z)/(max_z-min_z)
    pd.DataFrame([tuple(point) for point in all_points]).to_csv('3D_points.csv')

    data_frame = pd.read_csv('3D_points.csv')
    all_x = np.array(data_frame['0'])
    all_y = np.array(data_frame['1'])
    all_z = np.array(data_frame['2'])
    all_points = [[all_x[i], all_y[i], all_z[i]] for i in range(len(all_x))]
    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=3, random_state=0).fit(all_points)
    pred = kmeans.labels_
    center = kmeans.cluster_centers_
    print(pred)
    dataframe = pd.DataFrame({'path': all_path, 'label': pred})
    dataframe.to_csv('cluster.csv', index=False)


