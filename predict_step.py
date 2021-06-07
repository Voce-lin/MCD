import pandas as pd
import numpy as np
from train_step import get_all_filepath,load_data
from utils import utils,SG_anomaly_dector_1,SG_anomaly_dector_2,T_anomoly_detector,\
                  SR_anomoly_detector_1,SR_anomoly_detector_2,r_anomoly_detector,r_anomoly_detector_2


detectors=[SR_anomoly_detector_1(),
           SR_anomoly_detector_2(),
           SG_anomaly_dector_1(),
           SG_anomaly_dector_2(),
           r_anomoly_detector(),
           r_anomoly_detector_2(),
           T_anomoly_detector(),
           ]

detector_index={'0':None,
                '1':None,
                '2':None,
                '3': None,}
if __name__=="__main__":
    data_frame=pd.read_csv('cluster.csv').dropna()
    cluster_ids=np.array(data_frame['label'],dtype=np.str)
    data_frame=pd.read_csv('3D_points.csv').dropna()
    all_x = np.array(data_frame['0'])
    all_y = np.array(data_frame['1'])
    all_z = np.array(data_frame['2'])
    all_points = [[all_x[i], all_y[i], all_z[i]] for i in range(len(all_x))]
    cen_value=[]
    for i in range(3):
        all_member=[all_points[j] for j in range(len(cluster_ids)) if int(cluster_ids[j])==i]
        meanx=np.mean([point[0] for point in all_member])
        meany=np.mean([point[1] for point in all_member])
        meanz=np.mean([point[2] for point in all_member])
        cen_value.append([meanx,meany,meanz])
    all_path = get_all_filepath()
    center_features = []
    for center in cen_value:
        l=[]
        for point in all_points:
            res = np.sqrt(np.sum(np.square(np.array([center[0], center[1], center[2]]) - np.array([point[0], point[1], point[2]]))))
            l.append(res)
        ind=l.index(min(l))
        print(ind)
        center_features.append(load_data(all_path[ind]))

    util=utils()

    for i in range(len(center_features)):
        max_score=0
        best_detector=None
        for detector in detectors:
            predict_point=detector(center_features[i][0])[0]
            f1,recall,precision=util.cal_f1score(predict_point,center_features[i][1])
            score=f1
            print(score,end=' , ')
            if score>max_score:
                max_score=score
                best_detector=detector
        print('\n')
        detector_index[str(i)]=best_detector

    print(detector_index)


    #---------------------start to detect all data------------
    recalls=[]
    precisions=[]
    f1s=[]
    import time
    start=time.perf_counter()
    for i in range(len(all_path)):
        data_frame=pd.read_csv(all_path[i]).dropna()
        value=np.array(data_frame['value'])
        # value=value-np.min(value)/(np.max(value)-np.min(value))
        try:
            label=np.array(data_frame['label'])
        except:
            label=np.array(data_frame['is_anomaly'])
        a_detector=detector_index[cluster_ids[i]]
        # a_detector=detectors[6]
        predict_point=a_detector(value)[0]
        f1,recall,precision=util.cal_f1score(predict_point,label)
        f1s.append(f1)
        recalls.append(recall)
        precisions.append(precision)
        print(f"No.{i} TS's score is {f1},{recall},{precision}"+' '+str(a_detector))
    end=time.perf_counter()
    print(f'{end-start} s')
    print(np.average(f1s))
    print('recall:',np.average(recalls))
    print('precision:',np.average(precisions))