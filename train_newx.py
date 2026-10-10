from tensorflow.keras.callbacks import Callback, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

import tensorflow as tf
import model_with_gcnx as model_with_gcn

import numpy as np
import model_def
import datetime
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from global_varx import num_PRED, train_set_file, val_set_file, test_set_file, la_grids, lo_grids,\
    model_file_dir, LIG_file_dir, radar_file_dir, record_file_dir, use_gpu
import model_def_trans
from data_generatorx import DataGenerator_CLA, PredictDataGenerator_CLA, PredictDataGenerator_StepDeep, DataGenerator_StepDeep
# tf.disable_v2_behavior()
global_val_sumETS_max = -1e10
global_val_sumETS_max_loss = 1e10
global_val_sumETS_max_epoch = -1
global_val_loss_min = 1e10
global_val_loss_min_ETS = -1e10
global_val_loss_min_epoch = -1
model_record_name = None

def POD(y_true, y_pred):
    ytrue = K.flatten(y_true)
    ypred = K.sigmoid(K.flatten(y_pred))
    ypred = K.round(ypred)
    true_positives = K.sum(ytrue * ypred)
    possible_positives = K.sum(ytrue)
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def FAR(y_true, y_pred):
    ytrue = K.flatten(y_true)
    ypred = K.sigmoid(K.flatten(y_pred))
    ypred = K.round(ypred)
    true_positives = K.sum(ytrue * ypred)
    predicted_positives = K.sum(ypred)
    precision = true_positives / (predicted_positives + K.epsilon())
    return 1 - precision

def TS(y_true, y_pred):
    ytrue = K.flatten(y_true)
    ypred = K.sigmoid(K.flatten(y_pred))
    ypred = K.round(ypred)
    N1 = K.sum(ytrue * ypred)
    N1pN2 = K.sum(ypred)
    N1pN3 = K.sum(ytrue)
    N2 = N1pN2 - N1
    N3 = N1pN3 - N1
    TS = N1 / (N1 + N2 + N3 + K.epsilon())
    return TS

def ETS(y_true, y_pred):
    ytrue = K.flatten(y_true)
    ypred = K.sigmoid(K.flatten(y_pred))
    ypred = K.round(ypred)
    N1 = K.sum(ytrue * ypred)
    N1pN2 = K.sum(ypred)
    N1pN3 = K.sum(ytrue)
    N2 = N1pN2 - N1
    N3 = N1pN3 - N1
    r = (N1+N2)*(N1+N3)/(la_grids*lo_grids)
    TS = (N1-r) / (N1 + N2 + N3 - r + K.epsilon())
    return TS
 
def binary_acc(y_true,y_pred):
    ypred = K.sigmoid(y_pred)
    return K.mean(K.equal(y_true, K.round(ypred)), axis=-1)

# def weighted_loss(y_true, y_pred):  # binary classification
#     positive_weight = 22.052808
#     # positive_weight = 18
#     # positive_weight = 50
#     ytrue = K.flatten(y_true)
#     ypred = K.flatten(y_pred)
#     return tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=ypred, labels=ytrue, pos_weight=positive_weight))
#     # return tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, targets=y_true, pos_weight=positive_weight))
# def weighted_loss(y_true, y_pred):
#     # y_true, y_pred shape: [batch, num_PRED, la_grids*lo_grids, 2]
#     positive_weight_light = 22.052808  # 闪电正样本权重
#     positive_weight_radar = 5.0        # 雷达正样本权重
#     loss_weight_light = 0.5            # 闪电损失权重
#     loss_weight_radar = 0.5            # 雷达损失权重

#     # 闪电损失
#     ytrue_light = K.flatten(y_true[..., 0])
#     ypred_light = K.flatten(y_pred[..., 0])
#     loss_light = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(
#         logits=ypred_light, labels=ytrue_light, pos_weight=positive_weight_light))
#     # 雷达损失
#     ytrue_radar = K.flatten(y_true[..., 1])
#     ypred_radar = K.flatten(y_pred[..., 1])
#     loss_radar = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(
#         logits=ypred_radar, labels=ytrue_radar, pos_weight=positive_weight_radar))
#     # 总损失加权求和
#     return loss_weight_light * loss_light + loss_weight_radar * loss_radar
# loss_weight_light = 0.8
# loss_weight_radar = 0.2
loss_weight_light = 0.3
loss_weight_radar = 0.7
def weighted_loss(y_true, y_pred):
    positive_weight_light = 22.052808
    positive_weight_radar = 4.0
    # 闪电损失
    ytrue_light = K.flatten(y_true[..., 0])
    ypred_light = K.flatten(y_pred[..., 0])
    loss_light = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(
        logits=ypred_light, labels=ytrue_light, pos_weight=positive_weight_light))
    # 雷达损失
    ytrue_radar = K.flatten(y_true[..., 1])
    ypred_radar = K.flatten(y_pred[..., 1])
    loss_radar = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(
        logits=ypred_radar, labels=ytrue_radar, pos_weight=positive_weight_radar))
    return loss_weight_light * loss_light + loss_weight_radar * loss_radar
# def weighted_loss(y_true, y_pred):
#     # y_true, y_pred: (batch, num_PRED, nodes, 2), logits in y_pred
#     global loss_weight_light, loss_weight_radar
#     y_true = tf.cast(y_true, tf.float32)
#     y_pred = tf.cast(y_pred, tf.float32)

#     # hyperparams
#     pos_weight_light = 22.052808  # 闪电正样本权重（可保留）
#     radar_influence = 1.0         # radar 对 lightning 的放大倍数（gamma）
#     pos_weight_radar = 4.0        # 雷达正样本权重（原来值）

#     # ---------- lightning loss (sample-weighted BCE) ----------
#     y_l_true = y_true[..., 0]     # 0/1
#     y_l_logit = y_pred[..., 0]

#     # per-element BCE (logits form)
#     bce_per = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_l_true, logits=y_l_logit)

#     # sample-wise weight: base 1.0, 如果 radar==1 则加权
#     radar_flag = y_true[..., 1]
#     sample_weight = 1.0 + radar_influence * radar_flag

#     # 仍对正样本额外放大（将 pos_weight 转换为对正样本的乘子）
#     sample_weight = sample_weight + (pos_weight_light - 1.0) * y_l_true

#     # 归一化求和以避免 batch size 影响
#     loss_light = tf.reduce_sum(sample_weight * bce_per) / (tf.reduce_sum(sample_weight) + 1e-6)

#     # ---------- radar loss (keep original weighted CE) ----------
#     y_r_true = y_true[..., 1]
#     y_r_logit = y_pred[..., 1]
#     # flatten for weighted_cross_entropy_with_logits
#     loss_radar = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(
#         labels=tf.reshape(y_r_true, [-1]), logits=tf.reshape(y_r_logit, [-1]), pos_weight=pos_weight_radar))

#     # combine with global weights (you already define loss_weight_light / radar externally)
#     return loss_weight_light * loss_light + loss_weight_radar * loss_radar
# class DynamicLossWeightCallback(Callback):
#     def __init__(self, max_epoch=30, start_light=0.3, end_light=0.8, final_radar_weight=0.2, fix_after_frac=0.6):
#         """
#         max_epoch: 线性增长所用的总 epoch（或参考值）
#         start_light: 起始的 light 权重（例如 0.2）
#         end_light: 线性增长结束的 light 权重（例如 0.8）
#         final_radar_weight: 在训练后段固定的 radar 权重（例如 0.1-0.2）
#         fix_after_frac: 在达到 max_epoch 的 fix_after_frac * max_epoch 之后开始固定 radar 权重
#         """
#         super().__init__()
#         self.max_epoch = max_epoch
#         self.start_light = start_light
#         self.end_light = end_light
#         self.final_radar_weight = float(final_radar_weight)
#         self.fix_epoch = int(max(1, max_epoch * fix_after_frac))

#     def on_epoch_begin(self, epoch, logs=None):
#         global loss_weight_light, loss_weight_radar
#         if epoch < self.fix_epoch:
#             # 线性增长（按 max_epoch 归一化）
#             frac = min(1.0, epoch / float(self.max_epoch))
#             loss_weight_light = float(self.start_light + (self.end_light - self.start_light) * frac)
#             loss_weight_radar = 1.0 - loss_weight_light
#         else:
#             # 固定 radar 权重为 final_radar_weight, light 权重为剩余部分
#             loss_weight_radar = float(self.final_radar_weight)
#             loss_weight_light = 1.0 - loss_weight_radar
#         print(f"[DynamicLossWeight] epoch={epoch} -> loss_weight_light={loss_weight_light:.3f}, loss_weight_radar={loss_weight_radar:.3f}")

def do_train(train_list, val_list, test_list, model_num):
    
    train_batch_size = 1
    val_batchsize = 1
    class_num = 2
    epochs_num = 25
    initial_epoch_num = 0
    init_learning_rate =0.00022932
    # init_learning_rate = 1e-4
    training_patience = 5
    

    
    if model_num == 11:
        train_gen = DataGenerator_CLA(train_list, train_batch_size, class_num, generator_type='train')
        val_gen = DataGenerator_CLA(val_list, val_batchsize, class_num, generator_type='val')
        val_gen_2 = PredictDataGenerator_CLA(val_list, val_batchsize)
        test_gen = PredictDataGenerator_CLA(test_list, val_batchsize)
        model = model_with_gcn.ST_GLNet00()

    elif model_num == 12:
        train_gen = DataGenerator_CLA(train_list, train_batch_size, class_num, generator_type='train')
        val_gen = DataGenerator_CLA(val_list, val_batchsize, class_num, generator_type='val')
        val_gen_2 = PredictDataGenerator_CLA(val_list, val_batchsize)
        test_gen = PredictDataGenerator_CLA(test_list, val_batchsize)
        model = model_with_gcn.ST_GLPANet01()
    elif model_num == 13:
        train_gen = DataGenerator_CLA(train_list, train_batch_size, class_num, generator_type='train')
        val_gen = DataGenerator_CLA(val_list, val_batchsize, class_num, generator_type='val')
        val_gen_2 = PredictDataGenerator_CLA(val_list, val_batchsize)
        test_gen = PredictDataGenerator_CLA(test_list, val_batchsize)
        model = model_with_gcn.LightNetPlus()
    elif model_num == 14:
        train_gen = DataGenerator_CLA(train_list, train_batch_size, class_num, generator_type='train')
        val_gen = DataGenerator_CLA(val_list, val_batchsize, class_num, generator_type='val')
        val_gen_2 = PredictDataGenerator_CLA(val_list, val_batchsize)
        test_gen = PredictDataGenerator_CLA(test_list, val_batchsize)
        model = model_with_gcn.ST_GLPANet()
    elif model_num == 15:
        train_gen = DataGenerator_CLA(train_list, train_batch_size, class_num, generator_type='train')
        val_gen = DataGenerator_CLA(val_list, val_batchsize, class_num, generator_type='val')
        val_gen_2 = PredictDataGenerator_CLA(val_list, val_batchsize)
        test_gen = PredictDataGenerator_CLA(test_list, val_batchsize)
        model = model_with_gcn.ConvGRU()
    elif model_num == 16:
        train_gen = DataGenerator_StepDeep(train_list, train_batch_size, class_num, generator_type='train')
        val_gen = DataGenerator_StepDeep(val_list, val_batchsize, class_num, generator_type='val')
        val_gen_2 = PredictDataGenerator_StepDeep(val_list, val_batchsize)
        test_gen = PredictDataGenerator_StepDeep(test_list, val_batchsize)
        model = model_with_gcn.StepDeep()
    elif model_num == 17:
        train_gen = DataGenerator_CLA(train_list, train_batch_size, class_num, generator_type='train')
        val_gen = DataGenerator_CLA(val_list, val_batchsize, class_num, generator_type='val')
        val_gen_2 = PredictDataGenerator_CLA(val_list, val_batchsize)
        test_gen = PredictDataGenerator_CLA(test_list, val_batchsize)
        model = model_def_trans.UQ_STNet_tran()
    elif model_num == 18:
        train_gen = DataGenerator_CLA(train_list, train_batch_size, class_num, generator_type='train')
        val_gen = DataGenerator_CLA(val_list, val_batchsize, class_num, generator_type='val')
        val_gen_2 = PredictDataGenerator_CLA(val_list, val_batchsize)
        test_gen = PredictDataGenerator_CLA(test_list, val_batchsize)
        model = model_with_gcn.ST_GLPANet10()
    adam = Adam(learning_rate=init_learning_rate, clipnorm=1.0)
    model.compile(
        loss=weighted_loss,
        optimizer=adam,
        metrics=[POD, FAR, TS, ETS, binary_acc])
    
    # model_file_name = "%s-{epoch:02d}.h5" % model.name
    
    val_label = np.zeros(shape=[len(val_list), num_PRED, 113 * 177, 2], dtype=np.float32)
    
    # checkpoint = ModelCheckpoint(model_file_dir + model_file_name, monitor='val_loss', verbose=1,
                                #  save_best_only=False, mode='min')
    
    print('generating val label...')
    for id, ddt_item in enumerate(val_list):
        ddt = datetime.datetime.strptime(ddt_item, '%Y%m%d%H%M')
        for hour_plus in range(num_PRED):
            dt = ddt + datetime.timedelta(hours=hour_plus)
            dt_item = dt.strftime('%Y%m%d%H%M')
            with open(LIG_file_dir + dt_item + '_truth') as tfile:
                y_test = np.array(tfile.readlines(), dtype=np.float32)
            y_test[y_test >= 1] = 1
            val_label[id, hour_plus, :, 0] = y_test
            try:
                cref = np.load(radar_file_dir + 'CREF/' + dt.strftime('%Y%m%d%H') + '.npy').ravel()
                et = np.load(radar_file_dir + 'ET/' + dt.strftime('%Y%m%d%H') + '.npy').ravel()
                vil = np.load(radar_file_dir + 'VIL/' + dt.strftime('%Y%m%d%H') + '.npy').ravel()
            # 各自阈值
                cref_thr, et_thr, vil_thr = 32, 7, 3  # 可根据实际调整
                radar_flag = ((cref >= cref_thr) | (et >= et_thr) | (vil >= vil_thr)).astype(np.float32)
                radar_flag[y_test == 1] = 1  # 发生闪电时雷达剧烈强制为1
                val_label[id, hour_plus, :, 1] = radar_flag
            except:
                #print('val    fffffffffffffffffffff')
                radar_flag = np.zeros_like(y_test)
                radar_flag[y_test == 1] = 1
                val_label[id, hour_plus, :, 1] = radar_flag
    np.save('val_label.npy', val_label)
    
    test_label = np.zeros(shape=[len(test_list), num_PRED, 113 * 177, 2], dtype=np.float32)
    print('generating test label...')
    for id, ddt_item in enumerate(test_list):
        ddt = datetime.datetime.strptime(ddt_item, '%Y%m%d%H%M')
        for hour_plus in range(num_PRED):
            dt = ddt + datetime.timedelta(hours=hour_plus)
            dt_item = dt.strftime('%Y%m%d%H%M')
            with open(LIG_file_dir + dt_item + '_truth') as tfile:
                y_test = np.array(tfile.readlines(), dtype=int)
            y_test[y_test >= 1] = 1
            test_label[id, hour_plus, :, 0] = y_test
            try:
                cref = np.load(radar_file_dir + 'CREF/' + dt.strftime('%Y%m%d%H') + '.npy').ravel()
                et = np.load(radar_file_dir + 'ET/' + dt.strftime('%Y%m%d%H') + '.npy').ravel()
                vil = np.load(radar_file_dir + 'VIL/' + dt.strftime('%Y%m%d%H') + '.npy').ravel()
            # 各自阈值
                cref_thr, et_thr, vil_thr = 32, 7, 3# 可根据实际调整
                radar_flag = ((cref >= cref_thr) | (et >= et_thr) | (vil >= vil_thr)).astype(np.float32)
                radar_flag[y_test == 1] = 1  # 发生闪电时雷达剧烈强制为1
                test_label[id, hour_plus, :, 1] = radar_flag
            except:
                #print('test    fffffffffffffffffffff')
                radar_flag = np.zeros_like(y_test)
                radar_flag[y_test == 1] = 1
                test_label[id, hour_plus, :, 1] = radar_flag
    global model_record_name

    model_record_name = model.name
    
    class RecordMetricsAfterEpoch(Callback):
        def on_epoch_end(self, epoch, logs={}):
            with open(record_file_dir + model_record_name + '.txt', 'a') as f:
                f.write('epoch %d:\r\n' % (epoch + 1))
                for key in ['loss', 'POD', 'FAR', 'TS', 'binary_acc', 'val_loss', 'val_POD', 'val_FAR', 'val_TS']:
                    f.write('%s: %f   ' % (key, logs[key]))
            #file_name = f"{model_file_dir}/ckpt/{model.name}-epoch-{epoch + 1:02d}.ckpt"
            file_name = f"{model_file_dir}/{model.name}-epoch-RADAR-4-0.1-{epoch + 1:02d}.h5"
            model.save_weights(file_name)
            print(f'Model weights saved to {file_name}')
                    
    class calVALmetrics(Callback):
        def on_epoch_end(self, epoch, logs=None):
            print('generating val data...')
            y_pred = self.model.predict_generator(val_gen_2, workers=5, verbose=1)
            y_pred = 1.0 / (1.0 + np.exp(-y_pred))
            y_pred = np.round(y_pred)
            y_pred = y_pred.reshape(len(val_list), num_PRED, 113 * 177, 2)
            y_true = val_label
            y_pred_sum = np.sum(y_pred, axis=1)
            y_true_sum = np.sum(y_true, axis=1)
            tot_light = np.zeros(4, dtype=np.float)
            tot_radar = np.zeros(4, dtype=np.float)
            for id in range(len(val_list)):
                ypred_l = y_pred_sum[id, :, 0]
                ytrue_l = y_true_sum[id, :, 0]
                ypred_r = y_pred_sum[id, :, 1]
                ytrue_r = y_true_sum[id, :, 1]
                tN_l = np.zeros(4, dtype=np.float)
                tN_r = np.zeros(4, dtype=np.float)
                tN_l[0] = np.sum((ypred_l > 0) & (ytrue_l > 0))
                tN_l[1] = np.sum((ypred_l > 0) & (ytrue_l < 1))
                tN_l[2] = np.sum((ypred_l < 1) & (ytrue_l > 0))
                tN_l[3] = np.sum((ypred_l < 1) & (ytrue_l < 1))
                tN_r[0] = np.sum((ypred_r > 0) & (ytrue_r > 0))
                tN_r[1] = np.sum((ypred_r > 0) & (ytrue_r < 1))
                tN_r[2] = np.sum((ypred_r < 1) & (ytrue_r > 0))
                tN_r[3] = np.sum((ypred_r < 1) & (ytrue_r < 1))
                tot_light += tN_l
                tot_radar += tN_r
            POD_l = tot_light[0] / (tot_light[0] + tot_light[2] + 10e-7)
            FAR_l = tot_light[1] / (tot_light[0] + tot_light[1] + 10e-7)
            TS_l = tot_light[0] / (tot_light[0] + tot_light[1] + tot_light[2] + 10e-7)
            R_l = (tot_light[0] + tot_light[1]) * (tot_light[0] + tot_light[2]) / (tot_light.sum())
            ETS_l = (tot_light[0] - R_l) / ((tot_light[0] + tot_light[1] + tot_light[2]) - R_l )
            POD_r = tot_radar[0] / (tot_radar[0] + tot_radar[2] + 10e-7)
            FAR_r = tot_radar[1] / (tot_radar[0] + tot_radar[1] + 10e-7)
            TS_r = tot_radar[0] / (tot_radar[0] + tot_radar[1] + tot_radar[2] + 1e-7)
            R_r = (tot_radar[0] + tot_radar[1]) * (tot_radar[0] + tot_radar[2]) / (tot_radar.sum())
            ETS_r = (tot_radar[0] - R_r) / ((tot_radar[0] + tot_radar[1] + tot_radar[2]) - R_r )
            global global_val_sumETS_max, global_val_sumETS_max_loss, global_val_sumETS_max_epoch, global_val_loss_min, global_val_loss_min_ETS, global_val_loss_min_epoch
            if ETS_l > global_val_sumETS_max:
                global_val_sumETS_max = ETS_l
                global_val_sumETS_max_loss = logs['val_loss']
                global_val_sumETS_max_epoch = epoch + 1
            if logs['val_loss'] < global_val_loss_min:
                global_val_loss_min = logs['val_loss']
                global_val_loss_min_ETS = ETS_l
                global_val_loss_min_epoch = epoch + 1
            with open(record_file_dir + model_record_name + '.txt', 'a') as f:
                f.write('val_light_POD: %f   ' % POD_l)
                f.write('val_light_FAR: %f   ' % FAR_l)
                f.write('val_light_TS: %f   ' % TS_l)
                f.write('val_light_ETS: %f   ' % ETS_l)
                f.write(' | val_radar_POD: %f   ' % POD_r)
                f.write('val_radar_FAR: %f   ' % FAR_r)
                f.write('val_radar_TS: %f   ' % TS_r)
                f.write('val_radar_ETS: %f   ' % ETS_r)
                f.write('\r\n')

    class calTestmetrics(Callback):
        def on_epoch_end(self, epoch, logs=None):
            print('generating test data...')
            y_pred = self.model.predict_generator(test_gen, workers=5, verbose=1)
            y_pred = 1.0 / (1.0 + np.exp(-y_pred))
            y_pred = np.round(y_pred)
            y_pred = y_pred.reshape(len(test_list), num_PRED, 113 * 177, 2)
            y_true = test_label
            y_pred_sum = np.sum(y_pred, axis=1)
            y_true_sum = np.sum(y_true, axis=1)
            tot_light = np.zeros(4, dtype=np.float)
            tot_radar = np.zeros(4, dtype=np.float)
            for id in range(len(test_list)):
                ypred_l = y_pred_sum[id, :, 0]
                ytrue_l = y_true_sum[id, :, 0]
                ypred_r = y_pred_sum[id, :, 1]
                ytrue_r = y_true_sum[id, :, 1]
                tN_l = np.zeros(4, dtype=np.float)
                tN_r = np.zeros(4, dtype=np.float)
                tN_l[0] = np.sum((ypred_l > 0) & (ytrue_l > 0))
                tN_l[1] = np.sum((ypred_l > 0) & (ytrue_l < 1))
                tN_l[2] = np.sum((ypred_l < 1) & (ytrue_l > 0))
                tN_l[3] = np.sum((ypred_l < 1) & (ytrue_l < 1))
                tN_r[0] = np.sum((ypred_r > 0) & (ytrue_r > 0))
                tN_r[1] = np.sum((ypred_r > 0) & (ytrue_r < 1))
                tN_r[2] = np.sum((ypred_r < 1) & (ytrue_r > 0))
                tN_r[3] = np.sum((ypred_r < 1) & (ytrue_r < 1))
                tot_light += tN_l
                tot_radar += tN_r
            POD_l = tot_light[0] / (tot_light[0] + tot_light[2] + 10e-7)
            FAR_l = tot_light[1] / (tot_light[0] + tot_light[1] + 10e-7)
            TS_l = tot_light[0] / (tot_light[0] + tot_light[1] + tot_light[2] + 10e-7)
            R_l = (tot_light[0] + tot_light[1]) * (tot_light[0] + tot_light[2]) / (tot_light.sum() + 10e-7)
            ETS_l = (tot_light[0] - R_l) / ((tot_light[0] + tot_light[1] + tot_light[2]) - R_l )
            POD_r = tot_radar[0] / (tot_radar[0] + tot_radar[2] + 10e-7)
            FAR_r = tot_radar[1] / (tot_radar[0] + tot_radar[1] + 10e-7)
            TS_r = tot_radar[0] / (tot_radar[0] + tot_radar[1] + tot_radar[2] + 10e-7)
            R_r = (tot_radar[0] + tot_radar[1]) * (tot_radar[0] + tot_radar[2]) / (tot_radar.sum() )
            ETS_r = (tot_radar[0] - R_r) / ((tot_radar[0] + tot_radar[1] + tot_radar[2]) - R_r )
            with open(record_file_dir + model_record_name + '.txt', 'a') as f:
                f.write('test_light_POD: %f   ' % POD_l)
                f.write('test_light_FAR: %f   ' % FAR_l)
                f.write('test_light_TS: %f   ' % TS_l)
                f.write('test_light_ETS: %f   ' % ETS_l)
                f.write(' | test_radar_POD: %f   ' % POD_r)
                f.write('test_radar_FAR: %f   ' % FAR_r)
                f.write('test_radar_TS: %f   ' % TS_r)
                f.write('test_radar_ETS: %f   ' % ETS_r)
                f.write('\r\n')
                
    RMAE = RecordMetricsAfterEpoch()
    CVALM = calVALmetrics()
    CTESTM = calTestmetrics()
    DECAY = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=training_patience, verbose=1, mode='min', min_delta=0.0001, cooldown=3,
                              min_lr=1e-6)
    # 自动保存val_loss最小的权重
    best_model_path = f"{model_file_dir}/{model.name}-best.h5"
    checkpoint_best = ModelCheckpoint(
        best_model_path,
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode='min'
    )

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs_num,
        initial_epoch=initial_epoch_num,
        workers=5,
        max_queue_size=10,
        callbacks=[RMAE, CVALM, CTESTM, DECAY, checkpoint_best]
        #callbacks=[RMAE, CVALM, CTESTM, DECAY, checkpoint_best, DynamicLossWeightCallback(epochs_num)]  # 加入checkpoint_best
    )

    # 训练结束后自动加载最佳权重
    print(f"Loading best model weights from {best_model_path}")
    model.load_weights(best_model_path)
    print("Best model loaded.")

if __name__ == '__main__':
    
    # model_num = 24
    model_num = 11

    if use_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''


    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.compat.v1.Session(config=config)


    train_list = []
    with open(train_set_file, 'r') as file:
        for line in file:
            train_list.append(line.rstrip('\n'))

    val_list = []
    with open(val_set_file, 'r') as file:
        for line in file:
            val_list.append(line.rstrip('\n'))

    test_list = []
    with open(test_set_file, 'r') as file:
        for line in file:
            test_list.append(line.rstrip('\n'))

    global_val_sumETS_max = -1e10
    global_val_sumETS_max_loss = 1e10
    global_val_sumETS_max_epoch = -1
    global_val_loss_min = 1e10
    global_val_loss_min_ETS = -1e10
    global_val_loss_min_epoch = -1

    do_train(train_list, val_list, test_list, model_num)

    with open(record_file_dir + model_record_name + '.txt', 'a') as f:
        f.write('global_val_loss_min: %f   ' % global_val_loss_min)
        f.write('global_val_loss_min_ETS: %f   ' % global_val_loss_min_ETS)
        f.write('global_val_loss_min_epoch: %d   ' % global_val_loss_min_epoch)
        f.write('\r\n')
        f.write('global_val_sumETS_max: %f   ' % global_val_sumETS_max)
        f.write('global_val_sumETS_max_loss: %f   ' % global_val_sumETS_max_loss)
        f.write('global_val_sumETS_max_epoch: %d   ' % global_val_sumETS_max_epoch)
        f.write('\r\n')