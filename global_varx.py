# 该文件定义了将在整个项目中使用的全局变量。

num_WRF = 6     # 用于一个预测案例的WRF模拟小时数。
num_LIG = 6     # 用于一个预测案例的闪电观测小时数。
num_AWS = 6     # 用于一个预测案例的AWS观测小时数。
num_radar = 6   # 用于一个预测案例的雷达参数小时数。
num_PRED = 6    # 预测小时数。
la_grids =113 # 纬度方向的网格间隔数。
lo_grids = 177  # 经度方向的网格间隔数。
dim_WRF = 37    # WRF模拟参数的数量。
dim_AWS = 3     # AWS观测参数的数量。
dim_radar = 3   # 雷达参数的数量。
use_gpu = True  # 是否使用GPU或CPU进行训练、验证和测试。
# train_set_file = './train.txt'     # 用于训练的时间段。2015.06-09 & 2016.05-09 & 2017.05-06
# val_set_file = './val.txt'     # 用于验证的时间段。2017.07
# test_set_file = './test.txt'          # 用于测试的时间段。2017.08-09
train_set_file = './train_periods.txt'     # 用于训练的时间段。2015.06-09 & 2016.05-09 & 2017.05-06
val_set_file = './val_periods.txt'     # 用于验证的时间段。2017.07
test_set_file = './test_periods_HM.txt'          # 用于测试的时间段。2017.08-09
model_file_dir = 'train_dir/models/'        # 训练好的模型保存在此目录中。
record_file_dir = 'train_dir/records/'      # 训练日志保存在此目录中。
result_file_dir = 'test_dir/results/'       # 预测结果保存在此目录中。
score_file_dir = 'test_dir/scores/'         # 测试分数保存在此目录中。
visualization_file_dir = 'test_dir/visualization/'  # 可视化结果保存在此目录中。
curve_file_dir = 'test_dir/curves/'         # 性能曲线保存在此目录中。
WRF_file_dir = 'data_dir/WRF/'              # WRF模拟数据（不包括WRF ncl）保存在此目录中。
WRF_ncl_file_dir = 'data_dir/WRF_ncl/'      # WRF ncl数据保存在此目录中。
LIG_file_dir = 'data_dir/LIG/'              # 闪电观测数据保存在此目录中。
radar_file_dir = 'data_dir/RADAR/'          # 雷达参数数据保存在此目录中。
AWS_file_dir = 'data_dir/AWS_copy/'              # AWS观测数据保存在此目录中。


# 以下变量用于数据预处理，可以忽略。

time_shift = 1
use_good_start = False

def get_time_period(dt):
    time = dt.strftime("%H:%M:%S")
    hour = int(time[0:2])
    if 0 <= hour < 6:
        nchour = '00'
    elif 6 <= hour < 12:
        nchour = '06'
    elif 12 <= hour < 18:
        nchour = '12'
    elif 18 <= hour <= 23:
        nchour = '18'
    else:
        print('error')
    delta_hour = hour - int(nchour)
    return nchour, delta_hour
