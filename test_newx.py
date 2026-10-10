# Test a trained model and calculate performance scores for the model.

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.python.keras import backend as K
import datetime
import numpy as np 
import scorex
from tensorflow.keras.models import load_model
from spektral.layers import GCNConv  # Add this import statement
from global_varx import dim_WRF, dim_AWS, num_LIG, num_AWS, num_PRED, use_good_start, test_set_file, model_file_dir,\
    time_shift, get_time_period, la_grids, lo_grids, result_file_dir, use_gpu
# import train
import model_def
import model_def_trans
import model_with_gcnx as model_with_gcn
# from model_def_trans import UQ_STNet_tran
from data_generatorx import DataGenerator_CLA, PredictDataGenerator_CLA, PredictDataGenerator_StepDeep, DataGenerator_StepDeep

def do_test(test_list, model, model_file, model_num):
    # --------------------------------------------------
    test_batch_size = 1

    if model_num == 11:
        test_gen = PredictDataGenerator_CLA(test_list, test_batch_size)

    elif model_num == 12:
        test_gen = PredictDataGenerator_StepDeep(test_list, test_batch_size)

    print('generating test data and predicting...')
    if model_num <= 14:
        y_pred = model.predict(test_gen, workers=3, verbose=1)  # [len(test_list),num_frames,112*176,1]
        ypred = y_pred
        ypred = 1.0 / (1.0 + np.exp(-ypred))  # if model for prediction doesn't include a sigmoid layer
        print('Prediction shape:', ypred.shape)
        with tf.device('/cpu:0'):
            for id, ddt_item in enumerate(test_list):
                ddt = datetime.datetime.strptime(ddt_item, '%Y%m%d%H%M')
                utc = ddt + datetime.timedelta(hours=-8)  # 北京时间转换成UTC时间
                ft = utc + datetime.timedelta(hours=(-6) * time_shift)  # 时间偏移
                nchour, delta_hour = get_time_period(ft)
                delta_hour += time_shift * 6
                for hour_plus in range(num_PRED):
                    ypred_i = ypred[id][hour_plus]
                    dt = ddt + datetime.timedelta(hours=hour_plus)
                    dt_item = dt.strftime('%Y%m%d%H%M')
                    result_path = result_file_dir + model_file + '/'
                    if not os.path.isdir(result_path):
                        os.makedirs(result_path)
                    with open(result_path + '%s_h%d' % (dt_item, hour_plus), 'w') as rfile:
                        for i in range(la_grids * lo_grids):
        # ypred_i[i] 是长度为2的数组
                            rfile.write('%f\r\n' % ypred_i[i][0])
# ...existing code...
                    print(dt_item)
    
if __name__ == "__main__":

    if use_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.compat.v1.Session(config=config)
    K.set_session(sess)

    test_list = []
    with open(test_set_file, 'r') as file:
        for line in file:  
            test_list.append(line.rstrip('\n'))
    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # 设置GPU内存增长（防止一次性占用所有内存）
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            # 限制GPU内存使用（使用80%的GPU内存）
            print(f"GPU设备已配置: {gpus[0]}")
        except RuntimeError as e:
            print(f"GPU配置错误: {e}")
    else:
        print("警告: 没有找到GPU设备，将使用CPU")
    
    model_num = 11  # 设置为 15 来运行 PR92 方法
    model_file = "ST_GLNet00-epoch-RADAR-4-0.1-24" # 为结果和评分定义一个名称

    model_file_path = model_file_dir + "/" + "ST_GLNet00-epoch-RADAR-4-0.1-24"  + ".h5"
    # model = model_with_gcn.ST_GLNet_Radar_LIG()
    model = model_with_gcn.ST_GLNet00()
    model.load_weights(model_file_path)

    # 对于 PR92，do_test 函数中的 model 参数不会被使用，可以传递 None
    do_test(test_list, model, model_file, model_num) 
    
    # 评分函数将使用 model_file 来查找 PR92 的结果
    scorex.eva(model_file, 0.5)
    scorex.eva_each_hour(model_file, 0.5)
    sess.close()