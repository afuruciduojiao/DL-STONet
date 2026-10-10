import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import RNN, Layer, InputSpec
from tensorflow.keras.layers import Input, Conv2D, TimeDistributed, ConvLSTM2D, Lambda, Conv1D, Concatenate, Reshape, Activation, Cropping3D, Conv2DTranspose, LeakyReLU, Multiply, Add, Permute, Softmax, Conv3D,LayerNormalization, MultiHeadAttention, Dense, Dropout
from tensorflow.keras import backend as K
from tensorflow.keras import activations
from tensorflow.keras import constraints
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
#from keras.utils import conv_utils, generic_utils, tf_utils
from global_varx import dim_AWS, num_PRED, use_good_start, dim_radar, la_grids, lo_grids, num_LIG, num_radar
from spektral.layers import GCNConv
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.layers.recurrent import _standardize_args
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Bidirectional, Lambda
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import BatchNormalization
from model.ConvGRU2D import ConvGRU2D
def dynamic_adj_matrix(node_features):
    """
    node_features: (batch*time, num_nodes, feature_dim)
    返回动态邻接矩阵 (batch*time, num_nodes, num_nodes)
    """
    Q = Dense(64)(node_features)  # (B, N, 64)
    K_ = Dense(64)(node_features) # (B, N, 64)
    attn = tf.matmul(Q, K_, transpose_b=True)  # (B, N, N)
    attn = tf.nn.softmax(attn, axis=-1)
    return attn
def cbam_block(x, ratio=8, name=None, temporal_consistency=True, consistency_scale=4.0, consistency_tau=0.1):#consistency_scale=10.0, consistency_tau=0.03):
    """
    支持4D和5D输入：
    - 4D: (B, H, W, C) 直接做CBAM
    - 5D: (B, T, H, W, C) 对每帧做CBAM（用TimeDistributed包裹空间卷积）
    新增：对于5D输入，可启用 temporal_consistency 门来抑制只出现在末帧的噪声。
    """
    channel = x.shape[-1]

    if len(x.shape) == 5:
        # Channel Attention (逐帧)
        avg_pool = tf.reduce_mean(x, axis=[2, 3], keepdims=True)  # (B, T, 1, 1, C)
        max_pool = tf.reduce_max(x, axis=[2, 3], keepdims=True)
        mlp1 = Dense(channel // ratio, activation='relu')
        mlp2 = Dense(channel, activation='sigmoid')
        scale_avg = mlp2(mlp1(avg_pool))
        scale_max = mlp2(mlp1(max_pool))
        scale = Add()([scale_avg, scale_max])
        ch_refined = Multiply()([x, scale])  # (B, T, H, W, C)

        if temporal_consistency:
            # 计算时序一致性门（按空间位置）
            # last_frame: (B, H, W, C), temporal_mean: (B, H, W, C)
            last_frame = x[:, -1, :, :, :]
            temporal_mean = tf.reduce_mean(x, axis=1)
            # 在通道维取平均作为不一致度度量 -> (B, H, W, 1)
            diff = tf.reduce_mean(tf.abs(last_frame - temporal_mean), axis=-1, keepdims=True)
            # gate: 大diff -> 接近0（抑制），小diff -> 接近1（保留）
            gate = tf.sigmoid(-consistency_scale * (diff - consistency_tau))  # (B, H, W, 1)
            # 扩展到时间维并广播到通道维 (B, 1, H, W, 1) -> 与 ch_refined 相乘
            gate = tf.expand_dims(gate, axis=1)
            ch_refined = Multiply()([ch_refined, gate])

        # Spatial Attention (逐帧)
        avg_pool = tf.reduce_mean(ch_refined, axis=-1, keepdims=True)  # (B, T, H, W, 1)
        max_pool = tf.reduce_max(ch_refined, axis=-1, keepdims=True)
        concat = Concatenate(axis=-1)([avg_pool, max_pool])  # (B, T, H, W, 2)
        spat = TimeDistributed(
            Conv2D(1, kernel_size=7, padding='same', activation='sigmoid')
        )(concat)
        out = Multiply()([ch_refined, spat])  # (B, T, H, W, C)
        return out

    # 4D输入（保持原实现）
    avg_pool = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
    max_pool = tf.reduce_max(x, axis=[1, 2], keepdims=True)
    mlp1 = Dense(channel // ratio, activation='relu')
    mlp2 = Dense(channel, activation='sigmoid')
    scale_avg = mlp2(mlp1(avg_pool))
    scale_max = mlp2(mlp1(max_pool))
    scale = Add()([scale_avg, scale_max])
    channel_attn = Multiply()([x, scale])
    avg_pool = tf.reduce_mean(channel_attn, axis=-1, keepdims=True)
    max_pool = tf.reduce_max(channel_attn, axis=-1, keepdims=True)
    concat = Concatenate(axis=-1)([avg_pool, max_pool])
    spatial_attn = Conv2D(1, kernel_size=7, padding='same', activation='sigmoid')(concat)
    out = Multiply()([channel_attn, spatial_attn])
    return out
def pyramid_pooling_block(x, pool_sizes, use_temporal=True, temporal_filters=None):
    """
    T-PPM (修正版)
    - 若use_temporal=True：在time维做Conv1D并保留逐时刻序列，不做全局均值与tile
    - 空间多尺度池化对每个时间帧独立进行
    x: (B, T, H, W, C)
    return: (B, T, H, W, C')
    """
    bs = tf.shape(x)[0]
    t = tf.shape(x)[1]
    h = tf.shape(x)[2]
    w = tf.shape(x)[3]
    c = tf.shape(x)[4]

    # (B*T, H, W, C)
    x_reshaped = tf.reshape(x, (-1, h, w, x.shape[-1]))
    concat_list = [x_reshaped]

    if use_temporal:
        # (B, H, W, T, C) -> (B*H*W, T, C)
        x_time = tf.transpose(x, perm=[0, 2, 3, 1, 4])
        x_time = tf.reshape(x_time, (bs * h * w, t, x.shape[-1]))
        filters = temporal_filters or (x.shape[-1] if x.shape[-1] is not None else 32)
        # 建议使用较小filters，且保持逐时刻序列
        x_time = Conv1D(filters, kernel_size=3, padding='same', activation='relu')(x_time)  # (B*H*W, T, F)
        # 还原到(B, T, H, W, F)
        x_time = tf.reshape(x_time, (bs, h, w, t, filters))
        x_time = tf.transpose(x_time, perm=[0, 3, 1, 2, 4])
        # 展平为(B*T, H, W, F)以便与空间分支拼接
        x_time = tf.reshape(x_time, (-1, h, w, filters))
        concat_list.append(x_time)

    # 空间多尺度（逐帧）
    for pool_size in pool_sizes:
        pooled = tf.keras.layers.AveragePooling2D(pool_size, padding='same')(x_reshaped)
        upsampled = tf.keras.layers.UpSampling2D(size=pool_size, interpolation='bilinear')(pooled)
        upsampled = tf.image.resize(upsampled, (h, w))
        concat_list.append(upsampled)

    concat = Concatenate()(concat_list)  # (B*T, H, W, C')
    new_c = concat.shape[-1]
    concat = tf.reshape(concat, (bs, t, h, w, new_c))
    return concat
def ST_TransformerLight():
    """
    改进版轻量时空 Transformer（面向显存 <20GB）
    主要改进：时间位置/局部 Conv 增强、dropout + LayerNorm、较小的 emb/heads、对两通道输出使用独立小头。
    输入输出接口与原函数一致，可直接替换。
    """
    encoder1_inputs = Input(shape=(None, la_grids, lo_grids, dim_radar), name='encoder1_inputs')
    encoder2_inputs = Input(shape=(None, la_grids, lo_grids, 1), name='encoder2_inputs')
    encoder3_inputs = Input(shape=(None, la_grids, lo_grids, dim_AWS), name='encoder3_inputs')
    decoder_inputs_ = Input(shape=(None, la_grids, lo_grids, 1), name='decoder_inputs')  # 保留接口

    # 轻量通道压缩
    e1 = TimeDistributed(Conv2D(filters=20, kernel_size=1, padding='same', activation='relu'))(encoder1_inputs)
    e2 = TimeDistributed(Conv2D(filters=6,  kernel_size=1, padding='same', activation='relu'))(encoder2_inputs)
    e3 = TimeDistributed(Conv2D(filters=26, kernel_size=1, padding='same', activation='relu'))(encoder3_inputs)
    concat = Concatenate(axis=-1)([e1, e2, e3])  # (bs, T, H, W, C~52)

    # 小的空间局部卷积以保留局部结构
    emb_dim = 48   # 若显存紧张可降到 32
    concat = TimeDistributed(Conv2D(filters=emb_dim, kernel_size=3, padding='same', activation='relu'))(concat)
    concat = TimeDistributed(BatchNormalization())(concat)
    concat = TimeDistributed(Activation('relu'))(concat)

    # (bs, T, H, W, emb_dim) -> 准备按节点并行做时间注意力
    num_nodes = la_grids * lo_grids
    # 把最后一维静态告知 Keras，避免 None 导致 initializer 错误
    x = Lambda(lambda z: tf.reshape(z, (tf.shape(z)[0], tf.shape(z)[1], num_nodes, emb_dim)),
               output_shape=(None, num_nodes, emb_dim))(concat)
    x = Lambda(lambda z: tf.transpose(z, perm=[0,2,1,3]),
               output_shape=(num_nodes, None, emb_dim))(x)  # (bs, num_nodes, T, emb_dim)
    x = Lambda(lambda z: tf.reshape(z, (tf.shape(z)[0]*tf.shape(z)[1], tf.shape(z)[2], emb_dim)),
               output_shape=(None, emb_dim))(x)  # (bs*num_nodes, T, emb_dim)

    # 轻量时间位置增强（Conv1D 做短期局部模式 + dropout）
    x_local = Conv1D(filters=emb_dim, kernel_size=3, padding='same', activation='relu')(x)
    x = Add()([x, x_local])
    x = LayerNormalization(epsilon=1e-6)(x)
    x = Dropout(0.1)(x)

    # Transformer block（时间轴上），heads/ff_dim 小
    mha_heads = 3
    ff_dim = 96
    # 注意：emb_dim 必须能被 mha_heads 整除用于 key_dim 计算
    attn = MultiHeadAttention(num_heads=mha_heads, key_dim=emb_dim // mha_heads, dropout=0.1)(x, x)
    x = Add()([x, attn])
    x = LayerNormalization(epsilon=1e-6)(x)

    ff = Dense(ff_dim, activation='relu')(x)
    ff = Dropout(0.1)(ff)
    ff = Dense(emb_dim)(ff)
    x = Add()([x, ff])
    x = LayerNormalization(epsilon=1e-6)(x)

    # 池化时间维，映射到 num_PRED * 2（两个通道）
    pool = tf.keras.layers.GlobalAveragePooling1D()(x)  # (bs*num_nodes, emb_dim)

    # 使用小 head 分别预测两个通道，便于分别调整权重
    head_lig = Dense(emb_dim//2, activation='relu')(pool)
    head_lig = Dropout(0.1)(head_lig)
    out_lig = Dense(num_PRED, activation=None, name='out_lig')(head_lig)  # logits for lightning

    head_radar = Dense(emb_dim//2, activation='relu')(pool)
    head_radar = Dropout(0.1)(head_radar)
    out_radar = Dense(num_PRED, activation=None, name='out_radar')(head_radar)  # logits for radar

    # 合并为 (bs*num_nodes, num_PRED, 2)
    out_concat = Concatenate(axis=-1)([out_lig, out_radar])  # (bs*num_nodes, num_PRED*2)
    out_dense = out_concat  # already shape (bs*num_nodes, num_PRED*2)

    # 恢复 batch 维并 reshape -> (bs, num_nodes, num_PRED, 2)
    batch_from_concat = Lambda(lambda z: tf.shape(z)[0], name='get_batch')(concat)
    out_reshaped = Lambda(lambda inputs: tf.reshape(inputs[0],
                                                    (tf.cast(inputs[1], tf.int32),
                                                     num_nodes,
                                                     num_PRED,
                                                     2)),
                         name='reshape_to_bs_numPred_nodes')([out_dense, batch_from_concat])

    # 转为 (bs, num_PRED, num_nodes, 2) 然后 flatten 到 (-1, la*lo, 2)
    out_perm = Lambda(lambda z: tf.transpose(z, perm=[0,2,1,3]), name='perm_to_time_first')(out_reshaped)
    out_time_space = Reshape((num_PRED, la_grids * lo_grids, 2), name='reshape_time_space')(out_perm)
    final_out = Reshape((-1, la_grids * lo_grids, 2), input_shape=(num_PRED, la_grids * lo_grids, 2), name='final_reshape')(out_time_space)

    return Model(inputs=[encoder1_inputs, encoder2_inputs, encoder3_inputs, decoder_inputs_],
                 outputs=[final_out],
                 name='ST_TransformerLight')
def ST_GLPANet():
    encoder1_inputs = Input(shape=(None, la_grids, lo_grids, dim_radar), name='encoder1_inputs')
    encoder2_inputs = Input(shape=(None, la_grids, lo_grids, 1), name='encoder2_inputs')
    encoder3_inputs = Input(shape=(None, la_grids, lo_grids, dim_AWS), name='encoder3_inputs')
    batch_size = 1
    time_steps = 6

    # 创建固定的邻接矩阵
    grid_shape = (la_grids, lo_grids)
    la, lo = grid_shape
    num_nodes = la * lo
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for i in range(la):
        for j in range(lo):
            idx = i * lo + j
            if i > 0:
                adj_matrix[idx, (i - 1) * lo + j] = 1
            if i < la - 1:
                adj_matrix[idx, (i + 1) * lo + j] = 1
            if j > 0:
                adj_matrix[idx, i * lo + (j - 1)] = 1
            if j < lo - 1:
                adj_matrix[idx, i * lo + (j + 1)] = 1
    degree = np.sum(adj_matrix, axis=1)
    degree[degree == 0] = 1
    D_inv_sqrt = np.diag(1.0 / np.sqrt(degree))
    adj_matrix = np.dot(np.dot(D_inv_sqrt, adj_matrix), D_inv_sqrt)
    adj_matrix = tf.constant(adj_matrix, dtype=tf.float32, name="adj_matrix")

    # encoder1 固定邻接矩阵
    gcn_input1 = tf.reshape(encoder1_inputs, (batch_size * time_steps, la_grids * lo_grids, dim_radar))
    gcn_output1 = GCNConv(128, activation=LeakyReLU(alpha=0.05))([gcn_input1, adj_matrix])
    gcn_output1 = GCNConv(128, activation=LeakyReLU(alpha=0.05))([gcn_output1, adj_matrix])
    gcn_output1 = tf.reshape(gcn_output1, (batch_size, time_steps, la_grids, lo_grids, 128))
    gcn_output1 = pyramid_pooling_block(gcn_output1, pool_sizes=[1, 2, 4, 8], use_temporal=True, temporal_filters=6)
    gcn_output1 = TimeDistributed(Conv2D(128, (1, 1), padding='same', activation=None))(gcn_output1)
    gcn_output1 = TimeDistributed(BatchNormalization())(gcn_output1)
    gcn_output1 = TimeDistributed(Activation('relu'))(gcn_output1)
    gcn_output1 = cbam_block(gcn_output1, name='gcn1_cbam')
    residual1 = gcn_output1[:, -1, :, :, :]
    residual1_proj = Conv2D(128, (1, 1), padding='same')(residual1)
    alpha1 = K.constant(0.25, dtype=tf.float32)
    encoder1_convlstm, h1, c1 = ConvLSTM2D(filters=128, kernel_size=(5, 5), return_state=True, padding='same', return_sequences=False, name='encoder1_convlstm')(gcn_output1)
    encoder1_convlstm = Add()([encoder1_convlstm, Lambda(lambda x: alpha1 * x)(residual1_proj)])

    # encoder2 固定邻接矩阵
    gcn_input2 = tf.reshape(encoder2_inputs, (batch_size * time_steps, la_grids * lo_grids, 1))
    gcn_output2 = GCNConv(128, activation=LeakyReLU(alpha=0.05))([gcn_input2, adj_matrix])
    gcn_output2 = GCNConv(128, activation=LeakyReLU(alpha=0.05))([gcn_output2, adj_matrix])
    gcn_output2 = tf.reshape(gcn_output2, (batch_size, time_steps, la_grids, lo_grids, 128))
    gcn_output2 = pyramid_pooling_block(gcn_output2, pool_sizes=[1, 2, 4, 8], use_temporal=False, temporal_filters=8)
    gcn_output2 = TimeDistributed(Conv2D(8, (1, 1), padding='same', activation=None))(gcn_output2)
    gcn_output2 = TimeDistributed(BatchNormalization())(gcn_output2)
    gcn_output2 = TimeDistributed(Activation('relu'))(gcn_output2)
    gcn_output2 = cbam_block(gcn_output2, name='gcn2_cbam', temporal_consistency=False)
    residual2 = gcn_output2[:, -1, :, :, :]
    residual2_proj = Conv2D(8, (1, 1), padding='same')(residual2)
    alpha2 = K.constant(0, dtype=tf.float32)
    encoder2_convlstm, h2, c2 = ConvLSTM2D(filters=8, kernel_size=(5, 5), return_state=True, padding='same', return_sequences=False, name='encoder2_convlstm')(gcn_output2)
    encoder2_convlstm = Add()([encoder2_convlstm, Lambda(lambda x: alpha2 * x)(residual2_proj)])

    # encoder3 固定邻接矩阵
    gcn_input3 = tf.reshape(encoder3_inputs, (batch_size * time_steps, la_grids * lo_grids, dim_AWS))
    gcn_output3 = GCNConv(128, activation=LeakyReLU(alpha=0.05))([gcn_input3, adj_matrix])
    gcn_output3 = GCNConv(128, activation=LeakyReLU(alpha=0.05))([gcn_output3, adj_matrix])
    gcn_output3 = tf.reshape(gcn_output3, (batch_size, time_steps, la_grids, lo_grids, 128))
    gcn_output3 = pyramid_pooling_block(gcn_output3, pool_sizes=[1, 2, 4, 8], use_temporal=False, temporal_filters=16)
    gcn_output3 = TimeDistributed(Conv2D(32, (1, 1), padding='same', activation=None))(gcn_output3)
    gcn_output3 = TimeDistributed(BatchNormalization())(gcn_output3)
    gcn_output3 = TimeDistributed(Activation('relu'))(gcn_output3)
    gcn_output3 = cbam_block(gcn_output3, name='gcn3_cbam', temporal_consistency=False)
    residual3 = gcn_output3[:, -1, :, :, :]
    residual3_proj = Conv2D(32, (1, 1), padding='same')(residual3)
    alpha3 = K.constant(0, dtype=tf.float32)
    encoder3_convlstm, h3, c3 = ConvLSTM2D(filters=32, kernel_size=(5, 5), return_state=True, padding='same', return_sequences=False, name='encoder3_convlstm')(gcn_output3)
    encoder3_convlstm = Add()([encoder3_convlstm, Lambda(lambda x: alpha3 * x)(residual3_proj)])

    # 编码器到解码器
    h1 = Conv2D(filters=32, kernel_size=(1, 1), padding="same", name='h1_conv2d', activation='relu')(h1)
    c1 = Conv2D(filters=32, kernel_size=(1, 1), padding="same", name='c1_conv2d', activation='relu')(c1)
    h2 = Conv2D(filters=16, kernel_size=(1, 1), padding="same", name='h2_conv2d', activation='relu')(h2)
    c2 = Conv2D(filters=16, kernel_size=(1, 1), padding="same", name='c2_conv2d', activation='relu')(c2)
    h3 = Conv2D(filters=16, kernel_size=(1, 1), padding="same", name='h3_conv2d', activation='relu')(h3)
    c3 = Conv2D(filters=16, kernel_size=(1, 1), padding="same", name='c3_conv2d', activation='relu')(c3)
    h = Concatenate(axis=-1)([h1, h2, h3])
    c = Concatenate(axis=-1)([c1, c2, c3])
    h = Conv2D(filters=64, kernel_size=(3, 3), strides=(4, 4), padding='same', activation='relu', name="h_conv2d")(h)
    c = Conv2D(filters=64, kernel_size=(3, 3), strides=(4, 4), padding='same', activation='relu', name="c_conv2d")(c)

    decoder_inputs_ = Input(shape=(None, la_grids, lo_grids, 1), name='decoder_inputs')
    decoder_inputs = decoder_inputs_
    if use_good_start:
        decoder_inputs = Lambda(lambda x: tf.math.pow(x, 1. / 3) * 4 - 3.5)(decoder_inputs)
        sigmoid = Activation('sigmoid')
        decoder_inputs = sigmoid(decoder_inputs)

    de_conv2d_1 = TimeDistributed(
        Conv2D(filters=8, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu'), name='de_conv2d_1')
    de_conv2d_2 = TimeDistributed(
        Conv2D(filters=16, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu'), name='de_conv2d_2')
    de_convlstm = ConvLSTM2D(filters=64, kernel_size=(5, 5), name='de_convlstm', padding='same', return_state=True, return_sequences=True)
    de_conv2dT_1 = TimeDistributed(
        Conv2DTranspose(filters=64, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu'), name='de_conv2dT_1')
    de_conv2dT_2 = TimeDistributed(
        Conv2DTranspose(filters=64, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu'), name='de_conv2dT_2')
    de_out_conv2d = TimeDistributed(Conv2D(filters=2, kernel_size=(1, 1), padding='same'), name='de_out_conv2d')

    cropper = Cropping3D(cropping=((0, 0), (1, 2), (1, 2)))

    decoder_outputs_list = []
    h_state, c_state = h, c
    for i in range(num_PRED):
        decoder_conv2d_1 = de_conv2d_1(decoder_inputs)
        decoder_conv2d_2 = de_conv2d_2(decoder_conv2d_1)
        decoder_convlstm, h_new, c_new = de_convlstm([decoder_conv2d_2, h_state, c_state])
        decoder_convlstm = cbam_block(decoder_convlstm, name=f'decoder_cbam_{i}')
        decoder_conv2dT_1 = de_conv2dT_1(decoder_convlstm)
        decoder_conv2dT_2 = de_conv2dT_2(decoder_conv2dT_1)
        decoder_out_conv2d = de_out_conv2d(decoder_conv2dT_2)
        decoder_output = cropper(decoder_out_conv2d)
        decoder_outputs_list.append(decoder_output)
        if use_good_start:
            decoder_output = sigmoid(decoder_output)
            decoder_inputs = decoder_output
        h_state, c_state = h_new, c_new
    decoder_outputs = Concatenate(axis=1)(decoder_outputs_list)
    decoder_outputs_ = Reshape((-1, la_grids * lo_grids, 2), input_shape=(-1, la_grids, lo_grids, 2))(decoder_outputs)
    return Model(inputs=[encoder1_inputs, encoder2_inputs, encoder3_inputs, decoder_inputs_], outputs=[decoder_outputs_], name='ST_GLPANet')
def ST_GLPANet01():
    encoder1_inputs = Input(shape=(None, la_grids, lo_grids, dim_radar), name='encoder1_inputs')
    encoder2_inputs = Input(shape=(None, la_grids, lo_grids, 1), name='encoder2_inputs')
    encoder3_inputs = Input(shape=(None, la_grids, lo_grids, dim_AWS), name='encoder3_inputs')
    batch_size = 1
    time_steps = 6

    # 创建固定的邻接矩阵
    grid_shape = (la_grids, lo_grids)
    la, lo = grid_shape
    num_nodes = la * lo
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for i in range(la):
        for j in range(lo):
            idx = i * lo + j
            if i > 0:
                adj_matrix[idx, (i - 1) * lo + j] = 1
            if i < la - 1:
                adj_matrix[idx, (i + 1) * lo + j] = 1
            if j > 0:
                adj_matrix[idx, i * lo + (j - 1)] = 1
            if j < lo - 1:
                adj_matrix[idx, i * lo + (j + 1)] = 1
    degree = np.sum(adj_matrix, axis=1)
    degree[degree == 0] = 1
    D_inv_sqrt = np.diag(1.0 / np.sqrt(degree))
    adj_matrix = np.dot(np.dot(D_inv_sqrt, adj_matrix), D_inv_sqrt)
    adj_matrix = tf.constant(adj_matrix, dtype=tf.float32, name="adj_matrix")

    # encoder1 固定邻接矩阵
    gcn_input1 = tf.reshape(encoder1_inputs, (batch_size * time_steps, la_grids * lo_grids, dim_radar))
    gcn_output1 = GCNConv(128, activation=LeakyReLU(alpha=0.05))([gcn_input1, adj_matrix])
    gcn_output1 = GCNConv(128, activation=LeakyReLU(alpha=0.05))([gcn_output1, adj_matrix])
    gcn_output1 = tf.reshape(gcn_output1, (batch_size, time_steps, la_grids, lo_grids, 128))
    #gcn_output1 = pyramid_pooling_block(gcn_output1, pool_sizes=[1, 2, 4, 8], use_temporal=True, temporal_filters=16)
    gcn_output1 = TimeDistributed(Conv2D(128, (1, 1), padding='same', activation=None))(gcn_output1)
    gcn_output1 = TimeDistributed(BatchNormalization())(gcn_output1)
    gcn_output1 = TimeDistributed(Activation('relu'))(gcn_output1)
    gcn_output1 = cbam_block(gcn_output1, name='gcn1_cbam')
    residual1 = gcn_output1[:, -1, :, :, :]
    residual1_proj = Conv2D(128, (1, 1), padding='same')(residual1)
    alpha1 = K.constant(0.25, dtype=tf.float32)
    encoder1_convlstm, h1, c1 = ConvLSTM2D(filters=128, kernel_size=(5, 5), return_state=True, padding='same', return_sequences=False, name='encoder1_convlstm')(gcn_output1)
    encoder1_convlstm = Add()([encoder1_convlstm, Lambda(lambda x: alpha1 * x)(residual1_proj)])

    # encoder2 固定邻接矩阵
    gcn_input2 = tf.reshape(encoder2_inputs, (batch_size * time_steps, la_grids * lo_grids, 1))
    gcn_output2 = GCNConv(128, activation=LeakyReLU(alpha=0.05))([gcn_input2, adj_matrix])
    gcn_output2 = GCNConv(128, activation=LeakyReLU(alpha=0.05))([gcn_output2, adj_matrix])
    gcn_output2 = tf.reshape(gcn_output2, (batch_size, time_steps, la_grids, lo_grids, 128))
    #gcn_output2 = pyramid_pooling_block(gcn_output2, pool_sizes=[1, 2, 4, 8], use_temporal=False, temporal_filters=8)
    gcn_output2 = TimeDistributed(Conv2D(8, (1, 1), padding='same', activation=None))(gcn_output2)
    gcn_output2 = TimeDistributed(BatchNormalization())(gcn_output2)
    gcn_output2 = TimeDistributed(Activation('relu'))(gcn_output2)
    gcn_output2 = cbam_block(gcn_output2, name='gcn2_cbam')
    residual2 = gcn_output2[:, -1, :, :, :]
    residual2_proj = Conv2D(8, (1, 1), padding='same')(residual2)
    alpha2 = K.constant(0.5, dtype=tf.float32)
    encoder2_convlstm, h2, c2 = ConvLSTM2D(filters=8, kernel_size=(5, 5), return_state=True, padding='same', return_sequences=False, name='encoder2_convlstm')(gcn_output2)
    encoder2_convlstm = Add()([encoder2_convlstm, Lambda(lambda x: alpha2 * x)(residual2_proj)])

    # encoder3 固定邻接矩阵
    gcn_input3 = tf.reshape(encoder3_inputs, (batch_size * time_steps, la_grids * lo_grids, dim_AWS))
    gcn_output3 = GCNConv(128, activation=LeakyReLU(alpha=0.05))([gcn_input3, adj_matrix])
    gcn_output3 = GCNConv(128, activation=LeakyReLU(alpha=0.05))([gcn_output3, adj_matrix])
    gcn_output3 = tf.reshape(gcn_output3, (batch_size, time_steps, la_grids, lo_grids, 128))
    #gcn_output3 = pyramid_pooling_block(gcn_output3, pool_sizes=[1, 2, 4, 8], use_temporal=False, temporal_filters=16)
    gcn_output3 = TimeDistributed(Conv2D(32, (1, 1), padding='same', activation=None))(gcn_output3)
    gcn_output3 = TimeDistributed(BatchNormalization())(gcn_output3)
    gcn_output3 = TimeDistributed(Activation('relu'))(gcn_output3)
    gcn_output3 = cbam_block(gcn_output3, name='gcn3_cbam')
    residual3 = gcn_output3[:, -1, :, :, :]
    residual3_proj = Conv2D(32, (1, 1), padding='same')(residual3)
    alpha3 = K.constant(0.5, dtype=tf.float32)
    encoder3_convlstm, h3, c3 = ConvLSTM2D(filters=32, kernel_size=(5, 5), return_state=True, padding='same', return_sequences=False, name='encoder3_convlstm')(gcn_output3)
    encoder3_convlstm = Add()([encoder3_convlstm, Lambda(lambda x: alpha3 * x)(residual3_proj)])

    # 编码器到解码器
    h1 = Conv2D(filters=32, kernel_size=(1, 1), padding="same", name='h1_conv2d', activation='relu')(h1)
    c1 = Conv2D(filters=32, kernel_size=(1, 1), padding="same", name='c1_conv2d', activation='relu')(c1)
    h2 = Conv2D(filters=16, kernel_size=(1, 1), padding="same", name='h2_conv2d', activation='relu')(h2)
    c2 = Conv2D(filters=16, kernel_size=(1, 1), padding="same", name='c2_conv2d', activation='relu')(c2)
    h3 = Conv2D(filters=16, kernel_size=(1, 1), padding="same", name='h3_conv2d', activation='relu')(h3)
    c3 = Conv2D(filters=16, kernel_size=(1, 1), padding="same", name='c3_conv2d', activation='relu')(c3)
    h = Concatenate(axis=-1)([h1, h2, h3])
    c = Concatenate(axis=-1)([c1, c2, c3])
    h = Conv2D(filters=64, kernel_size=(3, 3), strides=(4, 4), padding='same', activation='relu', name="h_conv2d")(h)
    c = Conv2D(filters=64, kernel_size=(3, 3), strides=(4, 4), padding='same', activation='relu', name="c_conv2d")(c)

    decoder_inputs_ = Input(shape=(None, la_grids, lo_grids, 1), name='decoder_inputs')
    decoder_inputs = decoder_inputs_
    if use_good_start:
        decoder_inputs = Lambda(lambda x: tf.math.pow(x, 1. / 3) * 4 - 3.5)(decoder_inputs)
        sigmoid = Activation('sigmoid')
        decoder_inputs = sigmoid(decoder_inputs)

    de_conv2d_1 = TimeDistributed(
        Conv2D(filters=8, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu'), name='de_conv2d_1')
    de_conv2d_2 = TimeDistributed(
        Conv2D(filters=16, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu'), name='de_conv2d_2')
    de_convlstm = ConvLSTM2D(filters=64, kernel_size=(5, 5), name='de_convlstm', padding='same', return_state=True, return_sequences=True)
    de_conv2dT_1 = TimeDistributed(
        Conv2DTranspose(filters=64, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu'), name='de_conv2dT_1')
    de_conv2dT_2 = TimeDistributed(
        Conv2DTranspose(filters=64, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu'), name='de_conv2dT_2')
    de_out_conv2d = TimeDistributed(Conv2D(filters=2, kernel_size=(1, 1), padding='same'), name='de_out_conv2d')

    cropper = Cropping3D(cropping=((0, 0), (1, 2), (1, 2)))

    decoder_outputs_list = []
    h_state, c_state = h, c
    for i in range(num_PRED):
        decoder_conv2d_1 = de_conv2d_1(decoder_inputs)
        decoder_conv2d_2 = de_conv2d_2(decoder_conv2d_1)
        decoder_convlstm, h_new, c_new = de_convlstm([decoder_conv2d_2, h_state, c_state])
        decoder_convlstm = cbam_block(decoder_convlstm, name=f'decoder_cbam_{i}')
        decoder_conv2dT_1 = de_conv2dT_1(decoder_convlstm)
        decoder_conv2dT_2 = de_conv2dT_2(decoder_conv2dT_1)
        decoder_out_conv2d = de_out_conv2d(decoder_conv2dT_2)
        decoder_output = cropper(decoder_out_conv2d)
        decoder_outputs_list.append(decoder_output)
        if use_good_start:
            decoder_output = sigmoid(decoder_output)
            decoder_inputs = decoder_output
        h_state, c_state = h_new, c_new
    decoder_outputs = Concatenate(axis=1)(decoder_outputs_list)
    decoder_outputs_ = Reshape((-1, la_grids * lo_grids, 2), input_shape=(-1, la_grids, lo_grids, 2))(decoder_outputs)
    return Model(inputs=[encoder1_inputs, encoder2_inputs, encoder3_inputs, decoder_inputs_], outputs=[decoder_outputs_], name='ST_GLPANet01')
def ST_GLPANet10():
    encoder1_inputs = Input(shape=(None, la_grids, lo_grids, dim_radar), name='encoder1_inputs')
    encoder2_inputs = Input(shape=(None, la_grids, lo_grids, 1), name='encoder2_inputs')
    encoder3_inputs = Input(shape=(None, la_grids, lo_grids, dim_AWS), name='encoder3_inputs')
    batch_size = 1
    time_steps = 6

    # 创建固定的邻接矩阵
    grid_shape = (la_grids, lo_grids)
    la, lo = grid_shape
    num_nodes = la * lo
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for i in range(la):
        for j in range(lo):
            idx = i * lo + j
            if i > 0:
                adj_matrix[idx, (i - 1) * lo + j] = 1
            if i < la - 1:
                adj_matrix[idx, (i + 1) * lo + j] = 1
            if j > 0:
                adj_matrix[idx, i * lo + (j - 1)] = 1
            if j < lo - 1:
                adj_matrix[idx, i * lo + (j + 1)] = 1
    degree = np.sum(adj_matrix, axis=1)
    degree[degree == 0] = 1
    D_inv_sqrt = np.diag(1.0 / np.sqrt(degree))
    adj_matrix = np.dot(np.dot(D_inv_sqrt, adj_matrix), D_inv_sqrt)
    adj_matrix = tf.constant(adj_matrix, dtype=tf.float32, name="adj_matrix")

    # encoder1 固定邻接矩阵
    gcn_input1 = tf.reshape(encoder1_inputs, (batch_size * time_steps, la_grids * lo_grids, dim_radar))
    gcn_output1 = GCNConv(128, activation=LeakyReLU(alpha=0.05))([gcn_input1, adj_matrix])
    gcn_output1 = GCNConv(128, activation=LeakyReLU(alpha=0.05))([gcn_output1, adj_matrix])
    gcn_output1 = tf.reshape(gcn_output1, (batch_size, time_steps, la_grids, lo_grids, 128))
    gcn_output1 = pyramid_pooling_block(gcn_output1, pool_sizes=[1, 2, 4, 8], use_temporal=True, temporal_filters=16)
    # gcn_output1 = TimeDistributed(Conv2D(128, (1, 1), padding='same', activation=None))(gcn_output1)
    # gcn_output1 = TimeDistributed(BatchNormalization())(gcn_output1)
    # gcn_output1 = TimeDistributed(Activation('relu'))(gcn_output1)
    # gcn_output1 = cbam_block(gcn_output1, name='gcn1_cbam')
    # residual1 = gcn_output1[:, -1, :, :, :]
    # residual1_proj = Conv2D(128, (1, 1), padding='same')(residual1)
    # alpha1 = K.constant(0.25, dtype=tf.float32)
    encoder1_convlstm, h1, c1 = ConvLSTM2D(filters=128, kernel_size=(5, 5), return_state=True, padding='same', return_sequences=False, name='encoder1_convlstm')(gcn_output1)
    #encoder1_convlstm = Add()([encoder1_convlstm, Lambda(lambda x: alpha1 * x)(residual1_proj)])

    # encoder2 固定邻接矩阵
    gcn_input2 = tf.reshape(encoder2_inputs, (batch_size * time_steps, la_grids * lo_grids, 1))
    gcn_output2 = GCNConv(128, activation=LeakyReLU(alpha=0.05))([gcn_input2, adj_matrix])
    gcn_output2 = GCNConv(128, activation=LeakyReLU(alpha=0.05))([gcn_output2, adj_matrix])
    gcn_output2 = tf.reshape(gcn_output2, (batch_size, time_steps, la_grids, lo_grids, 128))
    gcn_output2 = pyramid_pooling_block(gcn_output2, pool_sizes=[1, 2, 4, 8], use_temporal=False, temporal_filters=8)
    # gcn_output2 = TimeDistributed(Conv2D(8, (1, 1), padding='same', activation=None))(gcn_output2)
    # gcn_output2 = TimeDistributed(BatchNormalization())(gcn_output2)
    # gcn_output2 = TimeDistributed(Activation('relu'))(gcn_output2)
    # gcn_output2 = cbam_block(gcn_output2, name='gcn2_cbam')
    # residual2 = gcn_output2[:, -1, :, :, :]
    # residual2_proj = Conv2D(8, (1, 1), padding='same')(residual2)
    # alpha2 = K.constant(0.5, dtype=tf.float32)
    encoder2_convlstm, h2, c2 = ConvLSTM2D(filters=8, kernel_size=(5, 5), return_state=True, padding='same', return_sequences=False, name='encoder2_convlstm')(gcn_output2)
    #encoder2_convlstm = Add()([encoder2_convlstm, Lambda(lambda x: alpha2 * x)(residual2_proj)])

    # encoder3 固定邻接矩阵
    gcn_input3 = tf.reshape(encoder3_inputs, (batch_size * time_steps, la_grids * lo_grids, dim_AWS))
    gcn_output3 = GCNConv(128, activation=LeakyReLU(alpha=0.05))([gcn_input3, adj_matrix])
    gcn_output3 = GCNConv(128, activation=LeakyReLU(alpha=0.05))([gcn_output3, adj_matrix])
    gcn_output3 = tf.reshape(gcn_output3, (batch_size, time_steps, la_grids, lo_grids, 128))
    gcn_output3 = pyramid_pooling_block(gcn_output3, pool_sizes=[1, 2, 4, 8], use_temporal=False, temporal_filters=16)
    # gcn_output3 = TimeDistributed(Conv2D(32, (1, 1), padding='same', activation=None))(gcn_output3)
    # gcn_output3 = TimeDistributed(BatchNormalization())(gcn_output3)
    # gcn_output3 = TimeDistributed(Activation('relu'))(gcn_output3)
    # gcn_output3 = cbam_block(gcn_output3, name='gcn3_cbam')
    # residual3 = gcn_output3[:, -1, :, :, :]
    # residual3_proj = Conv2D(32, (1, 1), padding='same')(residual3)
    # alpha3 = K.constant(0.5, dtype=tf.float32)
    encoder3_convlstm, h3, c3 = ConvLSTM2D(filters=32, kernel_size=(5, 5), return_state=True, padding='same', return_sequences=False, name='encoder3_convlstm')(gcn_output3)
    #encoder3_convlstm = Add()([encoder3_convlstm, Lambda(lambda x: alpha3 * x)(residual3_proj)])

    # 编码器到解码器
    h1 = Conv2D(filters=32, kernel_size=(1, 1), padding="same", name='h1_conv2d', activation='relu')(h1)
    c1 = Conv2D(filters=32, kernel_size=(1, 1), padding="same", name='c1_conv2d', activation='relu')(c1)
    h2 = Conv2D(filters=16, kernel_size=(1, 1), padding="same", name='h2_conv2d', activation='relu')(h2)
    c2 = Conv2D(filters=16, kernel_size=(1, 1), padding="same", name='c2_conv2d', activation='relu')(c2)
    h3 = Conv2D(filters=16, kernel_size=(1, 1), padding="same", name='h3_conv2d', activation='relu')(h3)
    c3 = Conv2D(filters=16, kernel_size=(1, 1), padding="same", name='c3_conv2d', activation='relu')(c3)
    h = Concatenate(axis=-1)([h1, h2, h3])
    c = Concatenate(axis=-1)([c1, c2, c3])
    h = Conv2D(filters=64, kernel_size=(3, 3), strides=(4, 4), padding='same', activation='relu', name="h_conv2d")(h)
    c = Conv2D(filters=64, kernel_size=(3, 3), strides=(4, 4), padding='same', activation='relu', name="c_conv2d")(c)

    decoder_inputs_ = Input(shape=(None, la_grids, lo_grids, 1), name='decoder_inputs')
    decoder_inputs = decoder_inputs_
    if use_good_start:
        decoder_inputs = Lambda(lambda x: tf.math.pow(x, 1. / 3) * 4 - 3.5)(decoder_inputs)
        sigmoid = Activation('sigmoid')
        decoder_inputs = sigmoid(decoder_inputs)

    de_conv2d_1 = TimeDistributed(
        Conv2D(filters=8, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu'), name='de_conv2d_1')
    de_conv2d_2 = TimeDistributed(
        Conv2D(filters=16, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu'), name='de_conv2d_2')
    de_convlstm = ConvLSTM2D(filters=64, kernel_size=(5, 5), name='de_convlstm', padding='same', return_state=True, return_sequences=True)
    de_conv2dT_1 = TimeDistributed(
        Conv2DTranspose(filters=64, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu'), name='de_conv2dT_1')
    de_conv2dT_2 = TimeDistributed(
        Conv2DTranspose(filters=64, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu'), name='de_conv2dT_2')
    de_out_conv2d = TimeDistributed(Conv2D(filters=2, kernel_size=(1, 1), padding='same'), name='de_out_conv2d')

    cropper = Cropping3D(cropping=((0, 0), (1, 2), (1, 2)))

    decoder_outputs_list = []
    h_state, c_state = h, c
    for i in range(num_PRED):
        decoder_conv2d_1 = de_conv2d_1(decoder_inputs)
        decoder_conv2d_2 = de_conv2d_2(decoder_conv2d_1)
        decoder_convlstm, h_new, c_new = de_convlstm([decoder_conv2d_2, h_state, c_state])
        decoder_convlstm = cbam_block(decoder_convlstm, name=f'decoder_cbam_{i}')
        decoder_conv2dT_1 = de_conv2dT_1(decoder_convlstm)
        decoder_conv2dT_2 = de_conv2dT_2(decoder_conv2dT_1)
        decoder_out_conv2d = de_out_conv2d(decoder_conv2dT_2)
        decoder_output = cropper(decoder_out_conv2d)
        decoder_outputs_list.append(decoder_output)
        if use_good_start:
            decoder_output = sigmoid(decoder_output)
            decoder_inputs = decoder_output
        h_state, c_state = h_new, c_new
    decoder_outputs = Concatenate(axis=1)(decoder_outputs_list)
    decoder_outputs_ = Reshape((-1, la_grids * lo_grids, 2), input_shape=(-1, la_grids, lo_grids, 2))(decoder_outputs)
    return Model(inputs=[encoder1_inputs, encoder2_inputs, encoder3_inputs, decoder_inputs_], outputs=[decoder_outputs_], name='ST_GLPANet10')
def ST_GLNet00():
    # 输入
    encoder1_inputs = Input(shape=(None, la_grids, lo_grids, dim_radar), name='encoder1_inputs')
    encoder2_inputs = Input(shape=(None, la_grids, lo_grids, 1), name='encoder2_inputs')
    encoder3_inputs = Input(shape=(None, la_grids, lo_grids, dim_AWS), name='encoder3_inputs')
    
    # 创建固定的邻接矩阵
    grid_shape = (la_grids, lo_grids)
    la, lo = grid_shape
    num_nodes = la * lo
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for i in range(la):
        for j in range(lo):
            idx = i * lo + j
            if i > 0:  # 上
                adj_matrix[idx, (i - 1) * lo + j] = 1
            if i < la - 1:  # 下
                adj_matrix[idx, (i + 1) * lo + j] = 1
            if j > 0:  # 左
                adj_matrix[idx, i * lo + (j - 1)] = 1
            if j < lo - 1:  # 右
                adj_matrix[idx, i * lo + (j + 1)] = 1

    degree = np.sum(adj_matrix, axis=1)
    degree[degree == 0] = 1  # 避免除零
    D_inv_sqrt = np.diag(1.0 / np.sqrt(degree))
    adj_matrix = np.dot(np.dot(D_inv_sqrt, adj_matrix), D_inv_sqrt)
    adj_matrix = tf.constant(adj_matrix, dtype=tf.float32, name="adj_matrix")
    
    batch_size = 1
    time_steps = 6
    # encoder1 GCN->ConvLSTM
    gcn_input1 = tf.reshape(encoder1_inputs, (batch_size * time_steps, la_grids * lo_grids, dim_radar))
    gcn_output1 = GCNConv(128, activation=LeakyReLU(alpha=0.05))([gcn_input1, adj_matrix])
    gcn_output1 = GCNConv(128, activation=LeakyReLU(alpha=0.05))([gcn_output1, adj_matrix])
    gcn_output1 = tf.reshape(gcn_output1, (-1, time_steps, la_grids, lo_grids, 128))
    #gcn_output1 = cbam_block(gcn_output1, name='gcn1_cbam')
    encoder1_convlstm, h1, c1 = ConvLSTM2D(filters=128, kernel_size=(5, 5), return_state=True, padding='same', return_sequences=False, name='encoder1_convlstm')(gcn_output1)
    

    # encoder2 GCN->ConvLSTM
    gcn_input2 = tf.reshape(encoder2_inputs, (batch_size * time_steps, la_grids*lo_grids, 1))
    gcn_output2 = GCNConv(128, activation=LeakyReLU(alpha=0.05))([gcn_input2, adj_matrix])
    gcn_output2 = GCNConv(128, activation=LeakyReLU(alpha=0.05))([gcn_output2, adj_matrix])
    gcn_output2 = tf.reshape(gcn_output2, (batch_size, time_steps, la_grids, lo_grids, 128))
    #gcn_output2 = cbam_block(gcn_output2, name='gcn2_cbam')
    encoder2_convlstm, h2, c2 = ConvLSTM2D(filters=8, kernel_size=(5, 5), return_state=True, padding='same', return_sequences=True, name='encoder2_convlstm')(gcn_output2)
    
    # encoder3 GCN->ConvLSTM
    gcn_input3 = tf.reshape(encoder3_inputs, (batch_size * time_steps, la_grids*lo_grids, dim_AWS))
    gcn_output3 = GCNConv(128, activation=LeakyReLU(alpha=0.05))([gcn_input3, adj_matrix])
    gcn_output3 = GCNConv(128, activation=LeakyReLU(alpha=0.05))([gcn_output3, adj_matrix])
    gcn_output3 = tf.reshape(gcn_output3, (batch_size, time_steps, la_grids, lo_grids, 128))
    #gcn_output3 = cbam_block(gcn_output3, name='gcn3_cbam')
    encoder3_convlstm, h3, c3 = ConvLSTM2D(filters=32, kernel_size=(5, 5), return_state=True, padding='same', return_sequences=True, name='encoder3_convlstm')(gcn_output3)
    
    # 编码器到解码器
    h1 = Conv2D(filters=32, kernel_size=(1, 1), padding="same", name='h1_conv2d', activation='relu')(h1)
    c1 = Conv2D(filters=32, kernel_size=(1, 1), padding="same", name='c1_conv2d', activation='relu')(c1)
    h2 = Conv2D(filters=16, kernel_size=(1, 1), padding="same", name='h2_conv2d', activation='relu')(h2)
    c2 = Conv2D(filters=16, kernel_size=(1, 1), padding="same", name='c2_conv2d', activation='relu')(c2)
    h3 = Conv2D(filters=16, kernel_size=(1, 1), padding="same", name='h3_conv2d', activation='relu')(h3)
    c3 = Conv2D(filters=16, kernel_size=(1, 1), padding="same", name='c3_conv2d', activation='relu')(c3)
    h = Concatenate(axis=-1)([h1, h2, h3])
    c = Concatenate(axis=-1)([c1, c2, c3])
    
    print(h.shape, c.shape)
    
    # 下采样以匹配解码器输入的空间维度
    h = Conv2D(filters=64, kernel_size=(3, 3), strides=(4, 4), padding='same', activation='relu', name="h_conv2d")(h)
    c = Conv2D(filters=64, kernel_size=(3, 3), strides=(4, 4), padding='same', activation='relu', name="c_conv2d")(c)
    print(h.shape, c.shape)
    
    decoder_inputs_ = Input(shape=(None, la_grids, lo_grids, 1), name='decoder_inputs')
    decoder_inputs = decoder_inputs_
    if use_good_start:
        decoder_inputs = Lambda(lambda x: np.power(x, 1. / 3) * 4 - 3.5)(decoder_inputs)
        sigmoid = Activation('sigmoid')
        decoder_inputs = sigmoid(decoder_inputs)
    
    de_conv2d_1 = TimeDistributed(
        Conv2D(filters=8, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu'), name='de_conv2d_1')
    de_conv2d_2 = TimeDistributed(
        Conv2D(filters=16, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu'), name='de_conv2d_2')
    de_convlstm = ConvLSTM2D(filters=64, kernel_size=(5, 5), name='de_convlstm', padding='same', return_state=True, return_sequences=True)
    de_conv2dT_1 = TimeDistributed(
        Conv2DTranspose(filters=64, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu'), name='de_conv2dT_1')
    de_conv2dT_2 = TimeDistributed(
        Conv2DTranspose(filters=64, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu'), name='de_conv2dT_2')
    de_out_conv2d = TimeDistributed(Conv2D(filters=2, kernel_size=(1, 1), padding='same'), name='de_out_conv2d')  # 修改为2通道
    
    cropper = Cropping3D(cropping=((0, 0), (1, 2), (1, 2)))
    
    decoder_outputs_list = []
    for i in range(num_PRED):
        decoder_conv2d_1 = de_conv2d_1(decoder_inputs)
        decoder_conv2d_2 = de_conv2d_2(decoder_conv2d_1)
        decoder_convlstm, h, c = de_convlstm([decoder_conv2d_2, h, c])
        decoder_conv2dT_1 = de_conv2dT_1(decoder_convlstm)
        decoder_conv2dT_2 = de_conv2dT_2(decoder_conv2dT_1)
        decoder_out_conv2d = de_out_conv2d(decoder_conv2dT_2)
        decoder_output = cropper(decoder_out_conv2d)
        decoder_outputs_list.append(decoder_output)
        if use_good_start:
            decoder_output = sigmoid(decoder_output)
            decoder_inputs = decoder_output
    decoder_outputs = Concatenate(axis=1)(decoder_outputs_list)
    print(la_grids, lo_grids,' xxxxxxxxxxxxxxxxxxx ')
    decoder_outputs_ = Reshape((-1, la_grids * lo_grids, 2), input_shape=(-1, la_grids, lo_grids, 2))(decoder_outputs) 
    print(la_grids, lo_grids,' xxxxxxxxxxxxxxxxxxx ')# 修改为2通道
    return Model(inputs=[encoder1_inputs, encoder2_inputs, encoder3_inputs, decoder_inputs_], outputs=[decoder_outputs_], name='ST_GLNet00')

def LightNetPlus(): # encoder downsample twice, first convolute then concat
    encoder1_inputs = Input(shape=(None, la_grids, lo_grids, dim_radar), name='encoder1_inputs')  # (bs, 6, 159, 159, dim_WRF)
    # print('encoder1_inputs:', K.shape(encoder1_inputs))
    encoder1_conv2d_1 = TimeDistributed(
        Conv2D(filters=64, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu'),
        name='en1_conv2d_1')(encoder1_inputs)
    encoder1_conv2d_2 = TimeDistributed(
        Conv2D(filters=64, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu'),
        name='en1_conv2d_2')(encoder1_conv2d_1)
    encoder1_convlstm, h1, c1 = ConvLSTM2D(filters=128, kernel_size=(5, 5),
                                           return_state=True, padding='same', return_sequences=True,
                                           name='en1_convlstm')(encoder1_conv2d_2)
    # --------------------------------------------------------------------------------
    # encoder2: layers definition && data flow  --------------------------------------
    encoder2_inputs = Input(shape=(None, la_grids, lo_grids, 1), name='encoder2_inputs')  # (bs, 3, 159, 159, 1)
    # print('encoder2_inputs:', K.shape(encoder2_inputs))
    encoder2_conv2d_1 = TimeDistributed(Conv2D(filters=4, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu'),
                                      name='en2_conv2d_1')(encoder2_inputs)
    encoder2_conv2d_2 = TimeDistributed(Conv2D(filters=4, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu'),
                                      name='en2_conv2d_2')(encoder2_conv2d_1)
    encoder2_convlstm, h2, c2 = ConvLSTM2D(filters=8, kernel_size=(5, 5),
                                           return_state=True, padding='same', return_sequences=True,
                                           name='en2_convlstm')(encoder2_conv2d_2)
    # --------------------------------------------------------------------------------
    # encoder3: layers definition && data flow  --------------------------------------
    encoder3_inputs = Input(shape=(None, la_grids, lo_grids, dim_AWS), name='encoder3_inputs')
    # print('encoder3_inputs:', K.shape(encoder3_inputs))
    encoder3_conv2d_1 = TimeDistributed(Conv2D(filters=4, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu'),
                                      name='en3_conv2d_1')(encoder3_inputs)
    encoder3_conv2d_2 = TimeDistributed(Conv2D(filters=4, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu'),
                                      name='en3_conv2d_2')(encoder3_conv2d_1)
    encoder3_convlstm, h3, c3 = ConvLSTM2D(filters=32, kernel_size=(5, 5),
                                           return_state=True, padding='same', return_sequences=True,
                                           name='en3_convlstm')(encoder3_conv2d_2)
    # --------------------------------------------------------------------------------
    # encoder to decoder: layers definition && data flow  --------------------

    h1 = Conv2D(filters=32, kernel_size=(1, 1), padding="same", name='h1_conv2d', activation='relu')(h1)
    c1 = Conv2D(filters=32, kernel_size=(1, 1), padding="same", name='c1_conv2d', activation='relu')(c1)
    h2 = Conv2D(filters=16, kernel_size=(1, 1), padding="same", name='h2_conv2d', activation='relu')(h2)
    c2 = Conv2D(filters=16, kernel_size=(1, 1), padding="same", name='c2_conv2d', activation='relu')(c2)
    h3 = Conv2D(filters=16, kernel_size=(1, 1), padding="same", name='h3_conv2d', activation='relu')(h3)
    c3 = Conv2D(filters=16, kernel_size=(1, 1), padding="same", name='c3_conv2d', activation='relu')(c3)
    h = Concatenate(axis=-1)([h1, h2, h3])  # (bs,  40, 40, 48+16=64)
    c = Concatenate(axis=-1)([c1, c2, c3])  # (bs,  40, 40, 48+16=64)
    # --------------------------------------------------------------------------------

    # decoder: layers definition -------------------------------

    decoder_inputs_ = Input(shape=(None, la_grids, lo_grids, 1), name='decoder_inputs')
    decoder_inputs = decoder_inputs_
    if use_good_start:
        decoder_inputs = Lambda(lambda x: np.power(x, 1. / 3) * 4 - 3.5)(decoder_inputs)
        sigmoid = Activation('sigmoid')
        decoder_inputs = sigmoid(decoder_inputs)

    de_conv2d_1 = TimeDistributed(
        Conv2D(filters=8, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu'), name='de_conv2d_1')
    de_conv2d_2 = TimeDistributed(
        Conv2D(filters=16, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu'), name='de_conv2d_2')
    de_convlstm = ConvLSTM2D(filters=64, kernel_size=(5, 5), name='de_convlstm', padding='same', return_state=True, return_sequences=True)
    de_conv2dT_1 = TimeDistributed(
        Conv2DTranspose(filters=64, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu'),
        name='de_conv2dT_1')
    de_conv2dT_2 = TimeDistributed(
        Conv2DTranspose(filters=64, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu'),
        name='de_conv2dT_2')
    de_out_conv2d = TimeDistributed(Conv2D(filters=2, kernel_size=(1, 1), padding='same'),
                                    name='de_out_conv2d')
    # ----------------------------------------------------------
    # cropper = Cropping3D(cropping=((0, 0), (0, 1), (0, 1)))
    cropper = Cropping3D(cropping=((0, 0), (1, 2), (1, 2)))
    # decoder: data flow-----------------------------------------

    decoder_outputs_list = []
    for i in range(num_PRED):
        decoder_conv2d_1 = de_conv2d_1(decoder_inputs)
        decoder_conv2d_2 = de_conv2d_2(decoder_conv2d_1)
        decoder_convlstm, h, c = de_convlstm([decoder_conv2d_2, h, c])
        decoder_conv2dT_1 = de_conv2dT_1(decoder_convlstm)
        decoder_conv2dT_2 = de_conv2dT_2(decoder_conv2dT_1)
        decoder_out_conv2d = de_out_conv2d(decoder_conv2dT_2)  # (bs, 1, 160, 160, 1)
        decoder_output = cropper(decoder_out_conv2d)  # (bs, 1, 159, 159, 1)
        decoder_outputs_list.append(decoder_output)
        if use_good_start:
            decoder_output = sigmoid(decoder_output)
            decoder_inputs = decoder_output

    decoder_outputs = Concatenate(axis=1)(decoder_outputs_list)
    print(decoder_outputs.shape,' decoder_outputs shape ')
    print(num_PRED, la_grids, lo_grids,' xxxxxxxxxxxxxxxxxxx ')
    decoder_outputs_ = Reshape((-1, la_grids * lo_grids, 2), input_shape=(-1, la_grids, lo_grids, 2))(decoder_outputs) 
    print(num_PRED, la_grids, lo_grids,' xxxxxxxxxxxxxxxxxxx ')# 修改为2通道
    print(decoder_outputs_.shape,' decoder_outputs_ shape ')
    return Model(inputs=[encoder1_inputs, encoder2_inputs, encoder3_inputs, decoder_inputs_], outputs=[decoder_outputs_], name='LightNetPlus')
def ConvGRU():
    # 输入
    encoder1_inputs = Input(shape=(None, la_grids, lo_grids, dim_radar), name='encoder1_inputs')
    encoder2_inputs = Input(shape=(None, la_grids, lo_grids, 1), name='encoder2_inputs')
    encoder3_inputs = Input(shape=(None, la_grids, lo_grids, dim_AWS), name='encoder3_inputs')
    
    encoder1_conv2d_1 = TimeDistributed(
        Conv2D(filters=64, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu'),
        name='en1_conv2d_1')(encoder1_inputs)
    encoder1_conv2d_2 = TimeDistributed(
        Conv2D(filters=64, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu'),
        name='en1_conv2d_2')(encoder1_conv2d_1)
    encoder1_conv_gru, h1 = ConvGRU2D(filters=128, kernel_size=(5,5), 
                                      return_state=True, return_sequences=True,
                                      padding='same',
                                      name='en1_conv_gru')(encoder1_conv2d_2)
    
    encoder2_conv2d_1 = TimeDistributed(
        Conv2D(filters=4, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu'),
        name='en2_conv2d_1')(encoder2_inputs)
    encoder2_conv2d_2 = TimeDistributed(
        Conv2D(filters=4, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu'),
        name='en2_conv2d_2')(encoder2_conv2d_1)
    encoder2_conv_gru, h2 = ConvGRU2D(filters=8, kernel_size=(5,5), 
                                      return_state=True, return_sequences=True,
                                      padding='same', 
                                      name='en2_conv_gru')(encoder2_conv2d_2)
    
    encoder3_conv2d_1 = TimeDistributed(
        Conv2D(filters=64, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu'),
        name='en3_conv2d_1')(encoder3_inputs)
    encoder3_conv2d_2 = TimeDistributed(
        Conv2D(filters=64, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu'),
        name='en3_conv2d_2')(encoder3_conv2d_1)
    encoder3_conv_gru, h3 = ConvGRU2D(filters=32, kernel_size=(5,5), 
                                      return_state=True, return_sequences=True,
                                      padding='same',
                                      name='en3_conv_gru')(encoder3_conv2d_2)
    
    # 状态融合
    h1 = Conv2D(filters=32, kernel_size=(1,1), padding='same', activation='relu')(h1)
    h2 = Conv2D(filters=16, kernel_size=(1,1), padding='same', activation='relu')(h2)
    h3 = Conv2D(filters=16, kernel_size=(1,1), padding='same', activation='relu')(h3)
    h = Concatenate(axis=-1)([h1, h2, h3])
    
    decoder_inputs_ = Input(shape=(None, la_grids, lo_grids, 1), name='decoder_inputs')
    decoder_inputs = decoder_inputs_
    # if use_good_start:
    #     decoder_inputs = Lambda(lambda x: np.power(x, 1. / 3) * 4 - 3.5)(decoder_inputs)
    #     sigmoid = Activation('sigmoid')
    #     decoder_inputs = sigmoid(decoder_inputs)

    de_conv2d_1 = TimeDistributed(Conv2D(8, (5,5), strides=(2,2), padding='same', activation='relu'))
    de_conv2d_2 = TimeDistributed(Conv2D(16, (5,5), strides=(2,2), padding='same', activation='relu'))
    de_conv_gru = ConvGRU2D(filters=64, kernel_size=(5,5), return_state=True, return_sequences=True, padding='same')
    de_conv2dT_1 = TimeDistributed(Conv2DTranspose(64, (5,5), strides=(2,2), padding='same', activation='relu'))
    de_conv2dT_2 = TimeDistributed(Conv2DTranspose(64, (5,5), strides=(2,2), padding='same', activation='relu'))
    de_out_conv2d = TimeDistributed(Conv2D(2, (1,1), padding='same'))
    cropper = Cropping3D(cropping=((0,0), (1,2), (1,2)))
    # cropper = Cropping3D(cropping=((0,0), (0,0), (0,0)))
    
    decoder_outputs_list = []
    
    for i in range(num_PRED):
        decoder_conv2d_1 = de_conv2d_1(decoder_inputs)
        decoder_conv2d_2 = de_conv2d_2(decoder_conv2d_1)
        decoder_conv_gru, h = de_conv_gru(decoder_conv2d_2, initial_state=h)
        decoder_conv2dT_1 = de_conv2dT_1(decoder_conv_gru)
        decoder_conv2dT_2 = de_conv2dT_2(decoder_conv2dT_1)
        decoder_out_conv2d = de_out_conv2d(decoder_conv2dT_2)
        decoder_output = cropper(decoder_out_conv2d)
        # decoder_output = decoder_out_conv2d
        decoder_outputs_list.append(decoder_output)
        if use_good_start:
            decoder_output = sigmoid(decoder_output)
            decoder_inputs = decoder_output
            
    decoder_outputs = Concatenate(axis=1)(decoder_outputs_list)
    decoder_outputs_ = Reshape((-1, la_grids * lo_grids, 2), input_shape=(-1, la_grids, lo_grids, 2))(decoder_outputs)
    
    return Model([encoder1_inputs, encoder2_inputs, encoder3_inputs, decoder_inputs_], [decoder_outputs_], name='ConvGRU')
def StepDeep():
    AWS_inputs = Input(shape=(num_PRED, la_grids, lo_grids, dim_AWS), name='AWS_inputs')
    _LIG_inputs = Input(shape=(num_LIG, la_grids, lo_grids, 1), name='LIG_inputs')
    LIG_inputs = Permute((4, 2, 3, 1))(_LIG_inputs)
    _radar_inputs = Input(shape=(num_radar, la_grids, lo_grids, dim_radar), name='radar_inputs')
    radar_inputs = Permute((2, 3, 1, 4))(_radar_inputs)
    radar_inputs = Reshape((1, la_grids, lo_grids, dim_radar*num_radar))(radar_inputs)
    # radar_inputs = Reshape((1, la_grids, lo_grids, -1))(_radar_inputs)
    
    conv_1 = Conv3D(filters=64, kernel_size=(2, 1, 1), padding='same', name='conv3d_1')(AWS_inputs)
    conv_1 = Activation('relu')(conv_1)
    conv_2 = Conv3D(filters=64, kernel_size=(1, 3, 3), padding='same', name='conv3d_2')(conv_1)
    conv_2 = Activation('relu')(conv_2)
    conv_3 = Conv3D(filters=128, kernel_size=(2, 3, 3), padding='same', name='conv3d_3')(conv_2)
    conv_3 = Activation('relu')(conv_3)
    conv_4 = Conv3D(filters=64, kernel_size=(3, 1, 1), padding='same', name='conv3d_4')(conv_3)
    conv_4 = Activation('relu')(conv_4)
    conv_5 = Conv3D(filters=64, kernel_size=(1, 3, 3), padding='same', name='conv3d_5')(conv_4)
    conv_5 = Activation('relu')(conv_5)
    conv_6 = Conv3D(filters=64, kernel_size=(3, 3, 3), padding='same', name='conv3d_6')(conv_5)
    conv_6 = Activation('relu')(conv_6)
    steps = []
    
    for i in range(num_PRED):
        conv_6_i = Cropping3D(cropping=((i, num_PRED - i - 1), (0, 0), (0, 0)))(conv_6)   # (bs, 1, 159, 159, 64)
        conv2d_in = Concatenate(axis=-1)([LIG_inputs, conv_6_i, radar_inputs])                        # (bs, 1, 159, 159, 64+3)
        conv2d_in = Lambda(lambda x: K.squeeze(x, axis=1))(conv2d_in)  # (bs, 159, 159, 67)
        conv2d_1_i = Conv2D(filters=64, kernel_size=(7, 7), padding='same', name='conv2d_1_%d' % i)(conv2d_in)
        conv2d_1_i = Activation('relu')(conv2d_1_i)
        conv2d_2_i = Conv2D(filters=2, kernel_size=(7, 7), padding='same', name='conv2d_2_%d' % i)(conv2d_1_i)
        steps.append(conv2d_2_i)
        
    # conv_out = Concatenate(axis=1)(steps)  # (bs, 6, 159, 159, 1)
    conv_out = Concatenate(axis=1, name='concat_steps')(steps)  # (batch, num_PRED, la_grids, lo_grids, 1)
    outputs = Reshape((num_PRED, la_grids * lo_grids, 2), name='reshape_outputs')(conv_out)
    return Model([_radar_inputs, _LIG_inputs, AWS_inputs ], outputs, name='StepDeep')