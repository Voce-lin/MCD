import numpy as np
from numpy.fft import fft
import scipy.signal
import pywt,operator
from sklearn.metrics import f1_score,recall_score,precision_score
from scipy.signal import find_peaks

class utils():
    '''
    工具类：
    ------------
    SR_anomoly_detector: 谱残差异常检测器
    ------------
    cal_intensity: 时序波动强度检测算子
    ------------
    SG_anomaly_dector: SG滤波器异常检测器
    '''
    def __init__(self):
        pass

    def range_k(self,data, index, index_list, T):
        ks = []
        for i in index_list:
            try:
                k = np.abs(data[index] - data[i]) / T
                ks.append(k)
            except:
                pass
        if len(ks) == 0: return 0
        return min(ks)

    def T_detector(self,data):
        '''
        针对周期序列检测算法
        :param data: 一维时序数据 like numpy array
        :return:
        '''
        try:
            y=np.array(data)
            yf = abs(fft(y))  # 取绝对值
            yfnormlize = yf / len(y)  # 处理
            yfhalf = yfnormlize[:int(len(yfnormlize) / 2)]  # 由于对称性，只取一半区间
            xf = np.arange(len(y))  # 频率
            xhalf = xf[range(int(len(y) / 2))]  # 取一半区间
            f_pos = find_peaks(yfhalf, threshold=0.05)[0]
            combine = [(point, yfhalf[point]) for point in f_pos]
            combine.sort(key=operator.itemgetter(1), reverse=True)
            T = int(len(y) / combine[0][0])
        except:
            return [[]]
        new_data = np.zeros(len(data))
        bad_index = []
        uni_data=T * (data - np.min(data)) / (np.max(data) - np.min(data))
        for index in range(len(data)):
            if index - 2 * T-10 > 0:
                k1 = self.range_k(uni_data, index, [i for i in range(index - 2 * T-10, index - 2 * T + 10) if i not in bad_index],
                             T)
                k2 = self.range_k(uni_data, index, [i for i in range(index - 1 * T-10, index - 1 * T + 10) if i not in bad_index],
                             T)
                if k1 + k2 > 0.5:
                    new_data[index] = 1
                    bad_index.append(index)
        the_list = np.where(new_data == 1)
        return the_list

    def SR_anomoly_detector(self,data, theata):
        '''
        :param data:  一维时序数据 like numpy array
        :param theata: 检测敏感度参数
        :return: 异常点位下标 numpy array
        '''

        fft_out = np.fft.fft(data)
        log_abs_y = np.log(np.abs(fft_out))
        angle_y = np.angle(fft_out)

        average_log_abs_y = np.convolve(log_abs_y, [0.25 for i in range(4)], mode='same')  # 振幅谱均值滤波
        R_f = log_abs_y - average_log_abs_y
        com = [complex(R_f[i], angle_y[i]) for i in range(len(R_f))]
        S_R_ = np.exp(com)
        S_R = np.fft.ifft(S_R_)
        average_S_R, standerd_S_R = np.mean(S_R), np.std(S_R)
        # the_list=np.where(np.abs((S_R-average_S_R)/average_S_R)>0.15)
        the_list = np.array(np.where(np.abs(S_R - average_S_R) > theata * standerd_S_R)[0]) # 异常点位数组

        return [the_list]

    def SR_(self,data, theata):
        the_list=self.SR_anomoly_detector(data,theata=theata)
        a=the_list[0]+1
        if len(a) != 0 and a[-1] == len(data):
            a[-1] -= 1
        b=the_list[0]-1
        c=np.hstack([a,b,the_list[0]])
        return [c]

    def r_anomaly_detector(self,data,window=8):
        '''
        :param data:
        :return:
        '''
        label=np.zeros(len(data))
        jumpers=[]
        r_value=self.r_trans(data,window=window)
        std=np.std(r_value)
        mean=np.mean(r_value)
        for index in range(len(r_value)):
            if np.abs(r_value[index]-mean)>3*std:
                jumpers.append(index)
        for jump_index in jumpers:
            try:
             label[jump_index-15:jump_index+15]=1
            except:
                label[jump_index]=1
        the_list=np.where(label==1)
        return the_list


    def cal_wavelet(self,data):
        '''
        :param data: 一维时序数据 like numpy array
        :return: 大小波之差倍数
        '''
        ya5, yd5, yd4, yd3, yd2, yd1 = pywt.wavedec(data, 'db4', level=5)
        x1=np.mean(ya5)
        x2=np.mean([np.abs(yd1).mean(), np.abs(yd2).mean(), np.abs(yd3).mean(), np.abs(yd4).mean(), np.abs(yd5).mean()])
        return x1/x2


    def cal_intensity(self,data):
        '''
        :param data: 一维时序数据 like numpy array
        :return: 波动强度分数 float
        '''
        # mean= np.convolve(data, [0.02 for i in range(50)], mode='same')
        mean = scipy.signal.savgol_filter(data, 31, 1)
        D_value = np.sum(np.abs(data - mean)) / len(data)
        score = D_value / ((np.max(mean) - np.min(mean)) / 2)
        return score * 100

    def cal_T(self,data):
        '''
        小波方差主周期计算
        :param data:
        :return:
        '''
        y = np.linspace(1, 401, num=400)
        std = []
        for _ in y:
            coefs, f = pywt.cwt(data=data, wavelet='cmor1.5-1.0', sampling_period=1, scales=_)
            coefs = np.array(coefs[0])
            quad = np.sum(np.square(np.abs(coefs)) * (len(data) - 0) / len(data))
            std.append(quad)
        T=std.index(max(std))
        T=max(std)
        return T

    def SG_anomaly_dector(self,data,window_length,polyorder):
        '''
        :param data: 一维时序数据 like numpy array
        :param window_length: 滤波拟合窗口长度 奇数
        :param polyorder: 拟合阶数 smaller smoother
        :return:异常点位下标 numpy array
        '''
        indexs=np.where(data<0)[0]
        if len(indexs)!=0:
            data=data+2*(0-np.min(data))
        smooth_line=scipy.signal.savgol_filter(data, window_length, polyorder)
        the_list = np.array(np.where((0.8*np.array(smooth_line)>data) | (data>1.2*np.array(smooth_line)))[0])
        a=the_list+1
        if len(a)!=0 and a[-1] == len(data):
            a[-1] -= 1
        b=the_list-1
        c=np.hstack([a,b,the_list])
        return [c]

    def down_sampling_to_len(self,data,length=500):
        '''
        this is a unreliable algorithm,only be guaranteed to get a data whose length is close to param:length
        :param data:  一维时序数据 like numpy array
        :param length: 下采样后期望的长度 int
        :return:  长度为length的np array
                  #also mapped the sequence values to 0-1
        '''
        data_len=len(data)
        stride=int(data_len/length)
        new_data=data[::stride+1]
        # new_data=(new_data-np.min(new_data))/(np.max(new_data)-np.min(new_data))
        # ave=np.average(new_data)
        # new_data=new_data-ave
        return new_data

    def cal_f1score(self, the_list, label):
        '''
        :param the_list: 检测的异常点下标数组
        :param label:   标签数组
        :return: 准确率
        '''
        y_true=label
        y_pred=np.zeros(len(label))
        try:
            y_pred[the_list]=1
        except:
            pass
        f1=f1_score(y_true,y_pred,average='macro')
        recall_s=recall_score(y_true,y_pred,zero_division=1)
        precision=precision_score(y_true,y_pred,zero_division=1)
        return f1,recall_s,precision

    def r_trans(self,data, window=8):
        head = np.ones(window)
        tail = np.ones(window)
        head = head * data[0]
        tail = tail * data[-1]
        data = np.hstack((head, data, tail))
        r_value = np.array(data)
        for i in range(window, len(data) - window):
            x=np.sum(data[i:i+window])
            y=np.sum(data[i-window:i])
            if y==0:
                z=1
            else:
                z=x/y
            r_value[i] = z
        r_value = r_value[window:len(r_value) - window]
        index=np.where(np.abs(r_value)>2.5*np.abs(np.mean(r_value)))[0]
        r_value[index]=2.5*np.mean(r_value)
        return r_value


class SR_anomoly_detector_1():
    '''
    定制参数滤波器 下同
    '''
    def __init__(self):
        self.theata=3
        self.util = utils()

    def __call__(self, *args, **kwargs):
        '''
        :param args: 待检测一维时序数据，
        :param kwargs: 暂无
        :return: 异常点位下标数组 numpy array
        '''
        return self.util.SR_anomoly_detector(data=args[0],theata=self.theata)

class SR_anomoly_detector_2():
    def __init__(self):
        self.theata=8
        self.util = utils()

    def __call__(self, *args, **kwargs):
        return self.util.SR_(data=args[0],theata=self.theata)

class r_anomoly_detector():
    def __init__(self):
        self.util = utils()

    def __call__(self, *args, **kwargs):
        return self.util.r_anomaly_detector(data=args[0])

class r_anomoly_detector_2():
    def __init__(self):
        self.util = utils()

    def __call__(self, *args, **kwargs):
        return self.util.r_anomaly_detector(data=args[0],window=50)


class T_anomoly_detector():
    def __init__(self):
        self.util = utils()

    def __call__(self, *args, **kwargs):
        return self.util.T_detector(data=args[0],)

class SG_anomaly_dector_1():
    def __init__(self):
        self.window_length=101
        self.poly_order=2
        self.util = utils()

    def __call__(self, *args, **kwargs):
        return self.util.SG_anomaly_dector(data=args[0],
                                      window_length=self.window_length,
                                      polyorder=self.poly_order)

class SG_anomaly_dector_2():
    def __init__(self):
        self.window_length=51
        self.poly_order=5
        self.util=utils()

    def __call__(self, *args, **kwargs):
        return self.util.SG_anomaly_dector(data=args[0],
                                      window_length=self.window_length,
                                      polyorder=self.poly_order)



