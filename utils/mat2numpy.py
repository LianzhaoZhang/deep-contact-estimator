import os
import argparse
import glob
import sys
sys.path.append('.')
import numpy as np
import pandas as pd
import scipy.io as sio
# import matplotlib.pyplot as plt
import math
import yaml

# import lcm
# from lcm_types.python import contact_t, leg_control_data_lcmt, microstrain_lcmt
import time

def mat2numpy_one_seq(data_pth, save_pth):
    """
    Load data from .mat file and genearate numpy files without splitting into train/val/test.

    Inputs:
    - data_pth: path to mat data folder
    - save_pth: path to numpy saving directory.

    Data should be stored in .mat file, and contain:
    - q: joint encoder value (num0_data,12)
    - qd: joint angular velocity (num_data,12)
    - p: foot position from FK (num_data,12)
    - v: foot velocity from FK (num_data,12)
    
    - imu_acc: linear acceleration from imu (num_data,3)
    - imu_omega: angular velocity from imu (num_data,3)
    - contacts: contact data (num_data,4)
                contacts are stored as binary values, in the order
                of right_front, left_front, right_hind, left_hind.

                FRONT
                1 0  RIGHT
                3 2
                BACK

                Contact value will be treated as binary values in the
                order of contacts[0], contacts[1], contacts[2], contacts[3]
                and be converted to decimal value in this function.

                Ex. [1,0,0,1] -> 9
                    [0,1,1,0] -> 6

    - tau_est (optional): estimated control torque (num_data,12)
    - F (optional): ground reaction force

    Output:
    """

    for data_name in glob.glob(data_pth+'*'): 
        
        print("loading... ", data_name)

        # load data
        raw_data = sio.loadmat(data_name)

        contacts = raw_data['contacts']
        q = raw_data['q']
        p = raw_data['p']
        qd = raw_data['qd']
        v = raw_data['v']
        acc = raw_data['imu_acc']
        omega = raw_data['imu_omega']

        # tau_est = raw_data['tau_est']
        # F = raw_data['F']

        # concatenate current data. 
        data = np.concatenate((q,qd,acc,omega,p,v),axis=1)
        
        # convert labels from binary to decimal
        label = binary2decimal(contacts).reshape((-1,1)) 

        print("Saving data to: "+save_pth+os.path.splitext(os.path.basename(data_name))[0]+".npy")

        np.save(save_pth+os.path.splitext(os.path.basename(data_name))[0]+".npy",data)
        np.save(save_pth+os.path.splitext(os.path.basename(data_name))[0]+"_label.npy",label)

        print("Done!")

def csv2numpy_one_seq(data_path, save_path):
    """
    从CSV文件加载数据并生成numpy文件，不分割为训练/验证/测试集。
    
    输入:
    - data_path: CSV数据文件路径
    - save_path: numpy文件保存目录
    
    数据应包含:
    - 关节位置、速度和力矩 (每条腿3个关节)
    - IMU数据 (加速度和角速度)
    - 接触状态数据 (4条腿的接触状态)
    """
    print("Reading CSV file...")
    data_path = data_path + 'merged_data_raw.csv'
    df = pd.read_csv(data_path)
    
    # 添加序列ID列
    print("Processing time sequences...")
    df['time_diff'] = df['Time'].diff()
    sequence_starts = (df['time_diff'] > 0.002) | (df['time_diff'].isna())
    df['sequence_id'] = sequence_starts.cumsum()
    
    grouped = df.groupby(['sequence_id', 'Time'])
    
    # 初始化数据列表
    all_data = []
    all_labels = []
    
    print("Processing groups...")
    for (seq_id, time), group in grouped:
        if len(group) != 4:
            print(f"Warning: Sequence {seq_id}, Timestamp {time} has {len(group)} entries instead of 4")
            continue
            
        # 确保腿的顺序为 RF, LF, RH, LH
        group = group.sort_values('Leg_ID')
        
        # 提取特征
        positions = group[['HAA_position', 'HFE_position', 'KFE_position']].values.flatten()
        velocities = group[['HAA_velocity', 'HFE_velocity', 'KFE_velocity']].values.flatten()
        torques = group[['HAA_torque', 'HFE_torque', 'KFE_torque']].values.flatten()
        
        # 获取IMU数据
        imu_data = group[['IMU_linear_acceleration_x', 'IMU_linear_acceleration_y', 
                         'IMU_linear_acceleration_z', 'IMU_angular_velocity_x',
                         'IMU_angular_velocity_y', 'IMU_angular_velocity_z']].iloc[0].values
        
        # 获取参考接触状态
        contact_ref = group['Contact_State_Reference'].values
        
        # 合并特征
        features = np.concatenate([
            positions,      # 12个关节位置
            velocities,    # 12个关节速度
            torques,       # 12个关节力矩
            imu_data,      # 6个IMU数据
            contact_ref    # 4个参考接触状态
        ])
        
        all_data.append(features)
        
        # 处理标签
        contact_measured = group['Contact_State_Measured'].values.reshape(1, -1)
        label = binary2decimal(contact_measured)
        all_labels.append(label.item())
    
    # 转换为numpy数组
    all_data = np.array(all_data)
    all_labels = np.array(all_labels).reshape(-1, 1)
    
    # 生成文件名
    base_filename = "sequence_data"
    
    print(f"Saving data to: {os.path.join(save_path, base_filename)}.npy")
    
    # 保存数据
    np.save(os.path.join(save_path, f"{base_filename}.npy"), all_data)
    np.save(os.path.join(save_path, f"{base_filename}_label.npy"), all_labels)
    # Save to CSV file
    data_df = pd.DataFrame(all_data)
    data_df['label'] = all_labels
    data_df.to_csv(os.path.join(save_path, f"{base_filename}.csv"), index=False)
    
    print(f"Generated {len(all_data)} samples")
    print("Done!")

def mat2numpy_split(data_pth, save_pth, train_ratio=0.7, val_ratio=0.15):
    """
    Load data from .mat file, concatenate into numpy array, and save as train/val/test.
    Inputs:
    - data_pth: path to mat data folder
    - save_pth: path to numpy saving directory.
    - train_ratio: ratio of training data
    - val_ratio: ratio of validation data
    Data should be stored in .mat file, and contain:
    - q: joint encoder value (num_data,12)
    - qd: joint angular velocity (num_data,12)
    - p: foot position from FK (num_data,12)
    - v: foot velocity from FK (num_data,12)
    - imu_acc: linear acceleration from imu (num_data,3)
    - imu_omega: angular velocity from imu (num_data,3)
    - contacts: contact data (num_data,4)
                contacts are stored as binary values, in the order
                of right_front, left_front, right_hind, left_hind.
                FRONT
                1 0  RIGHT
                3 2
                BACK
                Contact value will be treated as binary values in the
                order of contacts[0], contacts[1], contacts[2], contacts[3]
                and be converted to decimal value in this function.
                Ex. [1,0,0,1] -> 9
                    [0,1,1,0] -> 6
                     
    - tau_est (optional): estimated control torque (num_data,12)
    - F (optional): ground reaction force

    Output:
    - 
    """

    num_features = 54    
    train_data = np.zeros((0,num_features))
    val_data = np.zeros((0,num_features))
    test_data = np.zeros((0,num_features))
    train_label = np.zeros((0,1))
    val_label = np.zeros((0,1))
    test_label = np.zeros((0,1))

    # for all dataset in the folder
    for data_name in glob.glob(data_pth+'*'): 
        
        print("loading... ", data_name)

        # load data
        raw_data = sio.loadmat(data_name)

        contacts = raw_data['contacts']
        q = raw_data['q']
        p = raw_data['p']
        qd = raw_data['qd']
        v = raw_data['v']
        acc = raw_data['imu_acc']
        omega = raw_data['imu_omega']

        # tau_est = raw_data['tau_est']
        # F = raw_data['F']
        
        # concatenate current data. First we try without GRF
        cur_data = np.concatenate((q,qd,acc,omega,p,v),axis=1)
        
        # separate data into train/val/test
        num_data = np.shape(q)[0]
        num_train = int(train_ratio*num_data)
        num_val = int(val_ratio*num_data)
        num_test = num_data-num_train-num_val
        cur_val = cur_data[:num_val,:]
        cur_test = cur_data[num_val:num_val+num_test,:]
        cur_train = cur_data[num_val+num_test:,:]

        # stack with all other sequences
        train_data = np.vstack((train_data,cur_train))
        val_data = np.vstack((val_data,cur_val))
        test_data = np.vstack((test_data,cur_test))

        
        # convert labels from binary to decimal
        cur_label = binary2decimal(contacts).reshape((-1,1))   

        # stack labels 
        val_label = np.vstack((val_label,cur_label[:num_val,:]))
        test_label = np.vstack((test_label,cur_label[num_val:num_val+num_test,:]))
        train_label = np.vstack((train_label,cur_label[num_val+num_test:,:]))

        # break
    train_label = train_label.reshape(-1,)
    val_label = val_label.reshape(-1,)
    test_label = test_label.reshape(-1,)
    
    print("Saving data...")
    
    np.save(save_pth+"train.npy",train_data)
    np.save(save_pth+"val.npy",val_data)
    np.save(save_pth+"test.npy",test_data)
    np.save(save_pth+"train_label.npy",train_label)
    np.save(save_pth+"val_label.npy",val_label)
    np.save(save_pth+"test_label.npy",test_label)

    print("Generated ", train_data.shape[0], " training data.")
    print("Generated ", val_data.shape[0], " validation data.")
    print("Generated ", test_data.shape[0], " test data.")
    
    print(train_data.shape[0])
    print(val_data.shape[0])
    print(test_data.shape[0])

    print("Done!")
    # return data
    
def csv2numpy_split(data_path, save_path, train_ratio=0.7, val_ratio=0.15):
    print("Reading CSV file...")
    data_path = data_path + 'merged_data_raw.csv'
    df = pd.read_csv(data_path)
    
    # 添加序列ID列
    print("Processing time sequences...")
    df['time_diff'] = df['Time'].diff()
    # 当时间差大于正常采样周期的2倍时认为是新的序列开始
    # 假设正常采样周期约为0.001秒，这里使用0.002秒作为阈值
    sequence_starts = (df['time_diff'] > 0.002) | (df['time_diff'].isna())
    df['sequence_id'] = sequence_starts.cumsum()
    
    grouped = df.groupby(['sequence_id', 'Time'])

    # 初始化存储所有重组后数据的列表
    all_data = []
    all_labels = []
    
    print("Processing groups...")
    print_flag = True
    count = 0
    for (seq_id, time), group in grouped:
        count += 1
        if len(group) != 4:  # 确保每个时间戳有4条腿的数据
            print(f"Warning: Sequence {seq_id}, Timestamp {time} has {len(group)} entries instead of 4")
            continue
            
        # 确保腿的顺序为 RF, LF, RH, LH
        group = group.sort_values('Leg_ID')
        
        # 提取特征
        positions = group[['HAA_position', 'HFE_position', 'KFE_position']].values.flatten()
        velocities = group[['HAA_velocity', 'HFE_velocity', 'KFE_velocity']].values.flatten()
        torques = group[['HAA_torque', 'HFE_torque', 'KFE_torque']].values.flatten()
        
        # 获取IMU数据(每个时间戳的4行数据都相同，取第一行即可)
        imu_data = group[['IMU_linear_acceleration_x', 'IMU_linear_acceleration_y', 
                            'IMU_linear_acceleration_z', 'IMU_angular_velocity_x',
                            'IMU_angular_velocity_y', 'IMU_angular_velocity_z']].iloc[0].values
        
        # 获取接触状态
        contact_ref = group['Contact_State_Reference'].values
        
        # 合并所有特征
        features = np.concatenate([
            positions,      # 12个关节位置
            velocities,    # 12个关节速度
            torques,       # 12个关节力矩
            imu_data,      # 6个IMU数据
            contact_ref   # 4个参考接触状态
        ])
        
        all_data.append(features)
        
        # 将测量的接触状态转换为二进制标签
        contact_measured = group['Contact_State_Measured'].values.reshape(1, -1)
        label = binary2decimal(contact_measured)
        
        all_labels.append(label.item())
        
        # if (print_flag and count == 1000):
        #     print_flag = False
        #     print("contact_measured shape:", contact_measured.shape)
        #     print("contact_measured values:", contact_measured)
        #     print("all_label shape:", len(all_labels))
        
    # 转换为numpy数组
    all_data = np.array(all_data)
    all_labels = np.array(all_labels)
    
    print("all_data shape:", all_data.shape)
    print("all_labels shape:", all_labels.shape)

    # 分割数据集
    num_samples = len(all_data)
    num_train = 73578
    num_val = 15766

    # 随机打乱数据
    indices = np.random.permutation(num_samples)
    all_data = all_data[indices]
    all_labels = all_labels[indices]

    # 分割数据集
    train_data = all_data[:num_train]
    val_data = all_data[num_train:num_train+num_val]
    test_data = all_data[num_train+num_val:]

    train_label = all_labels[:num_train]
    val_label = all_labels[num_train:num_train+num_val]
    test_label = all_labels[num_train+num_val:]
    
    # 保存数据
    print("Saving data...")
    np.save(os.path.join(save_path, "train.npy"), train_data)
    np.save(os.path.join(save_path, "val.npy"), val_data)
    np.save(os.path.join(save_path, "test.npy"), test_data)
    np.save(os.path.join(save_path, "train_label.npy"), train_label)
    np.save(os.path.join(save_path, "val_label.npy"), val_label)
    np.save(os.path.join(save_path, "test_label.npy"), test_label)
    # 保存为CSV文件
    train_df = pd.DataFrame(train_data)
    val_df = pd.DataFrame(val_data)
    test_df = pd.DataFrame(test_data)
    train_df['label'] = train_label
    val_df['label'] = val_label
    test_df['label'] = test_label
    train_df.to_csv(os.path.join(save_path, "train.csv"), index=False)
    val_df.to_csv(os.path.join(save_path, "val.csv"), index=False)
    test_df.to_csv(os.path.join(save_path, "test.csv"), index=False)

    print(f"Generated {len(train_data)} training samples")
    print(f"Generated {len(val_data)} validation samples")
    print(f"Generated {len(test_data)} test samples")
    print("Done!")

def binary2decimal(a, axis=-1):
    if a.ndim == 1:
        a = a.reshape(1, -1)
    return np.right_shift(np.packbits(a, axis=axis), 8 - a.shape[axis]).squeeze()

def mat2lcm(config):
    import lcm
    from lcm_types.python import contact_t, leg_control_data_lcmt, microstrain_lcmt
    mat_data = sio.loadmat(config['mat_data_path'])
    log = lcm.EventLog(config['lcm_save_path'], mode='w', overwrite=True)
    
    utime = int(time.time() * 10**6)

    imu_time = mat_data['imu_time'].flatten().tolist()
    
    for idx,_ in enumerate(imu_time[config['window_size']-1:]):

        data_idx = idx + config['window_size']-1
        
        leg_control_data_msg = leg_control_data_lcmt()
        leg_control_data_msg.q = mat_data['q'][data_idx]
        leg_control_data_msg.p = mat_data['p'][data_idx]
        leg_control_data_msg.qd = mat_data['qd'][data_idx]
        leg_control_data_msg.v = mat_data['v'][data_idx]
        leg_control_data_msg.tau_est = mat_data['tau_est'][data_idx]
        log.write_event(utime + int(10**6 * imu_time[data_idx]),\
                    'leg_control_data', leg_control_data_msg.encode())
        
        contact_msg = contact_t()
        contact_msg.num_legs = 4
        contact_msg.timestamp = imu_time[data_idx]
        if config['contact_type'] == 'GRF':
            contact_msg.contact = mat_data['F_contacts'][data_idx]
        elif config['contact_type'] == 'gait_cycle':
            contact_msg.contact = mat_data['gait_cycle_contacts'][data_idx]
        elif config['contact_type'] == 'contact':
            contact_msg.contact = mat_data['contacts'][data_idx]

        # if we want to use GT contact for varification
        # contact_msg.contact = mat_data['contacts'][data_idx]
        
        log.write_event(utime + int(10**6 * imu_time[data_idx]),\
                        'contact', contact_msg.encode())
        
        imu_msg = microstrain_lcmt()
        imu_msg.acc = mat_data['imu_acc'][data_idx]
        imu_msg.omega = mat_data['imu_omega'][data_idx]
        imu_msg.rpy = mat_data['imu_rpy'][data_idx]
        imu_msg.quat = mat_data['imu_quat'][data_idx]
        log.write_event(utime + int(10**6 * imu_time[data_idx]),\
                        'microstrain', imu_msg.encode())
        
    print("Saved data to lcm!")

def main():

    parser = argparse.ArgumentParser(description='Convert mat to numpy.')
    parser.add_argument('--config_name', type=str, default=os.path.dirname(os.path.abspath(__file__))+'/../config/mat2numpy_config.yaml')
    args = parser.parse_args()

    # config = yaml.load(open(args.config_name))
    with open(args.config_name, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    if config['mode']=='train':
        # mat2numpy_split(config['mat_folder'],config['save_path'],config['train_ratio'],config['val_ratio'])
        csv2numpy_split(config['csv_path'],config['csv_save_path'],config['train_ratio'],config['val_ratio'])
    elif config['mode']=='inference':
        # mat2numpy_one_seq(config['mat_folder'],config['save_path'])
        csv2numpy_one_seq(config['csv_path'],config['csv_save_path'])
    elif config['mode']=='mat2lcm':
        mat2lcm(config)
    

if __name__ == '__main__':
    main()
