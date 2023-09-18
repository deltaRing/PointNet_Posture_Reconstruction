import torch
import torch.nn.functional as F
import numpy as np

class MultiFittingLayer(torch.nn.Module):
    def __init__(self,
                 batch_size,
                 input_size,
                 hidden_size,
                 output_size):
        super(MultiFittingLayer, self).__init__()

        # 数据层大小
        self.batch_size = batch_size
        # 输入层大小 （pointnet++ 隐藏层）
        self.input_size = input_size
        # 中间层大小
        self.hidden_size = hidden_size
        # 输出层大小（均值、方差）
        self.output_size = output_size

        # 将数据压缩到 hidden_size x hidden_size
        self.fitting_model = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU(),  # fitting
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),  # self attention and non-linear
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),  # self attention and non-linear
        )

    def forward(self, x):
        return self.fitting_model(x)

class MultiConv1DLayer(torch.nn.Module):
    def __init__(self,
                 batch_size,
                 input_size,
                 hidden_size,
                 output_size):
        super(MultiConv1DLayer, self).__init__()

        # 数据层大小
        self.batch_size = batch_size
        # 输入层大小 （pointnet++ 隐藏层）
        self.input_size = input_size
        # 中间层大小
        self.hidden_size = hidden_size
        # 输出层大小（均值、方差）
        self.output_size = output_size

        # 输入: batch x 1 x hidden
        self.model = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(1, 32, 9, 2), # batch x hidden_size
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),
            torch.nn.ConvTranspose2d(32, 64, 9, 2),  # batch x hidden_size
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.ConvTranspose2d(64, 16, 9, 2),  # batch x hidden_size
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(16),
        )

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.view(self.batch_size,1,
                       int(np.sqrt(self.hidden_size)),
                       int(np.sqrt(self.hidden_size))) # batch_size x hidden_size x hidden_size
        return self.model(x)

class Self_Attention(torch.nn.Module):
    def __init__(self,
                 batch_size, # 数据批次大小
                 input_size, # 数据输入大小
                 poses):     # 数据种类
        super(Self_Attention, self).__init__()

        # 填充参数
        self.batch_size = batch_size
        # 输入参数
        self.input_size = input_size
        # 姿态通道
        self.poses      = poses

        # 五级的自注意力层
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(poses, poses, 3),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(poses),
            torch.nn.Conv2d(poses, poses, 3),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(poses),
            torch.nn.Conv2d(poses, poses, 3),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(poses),
            torch.nn.Conv2d(poses, poses, 3),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(poses),
            torch.nn.Conv2d(poses, poses, 3),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(poses),
        )

    def forward(self, x):
        return self.model(x)

class Auto_Point_Encoder(torch.nn.Module):
    def __init__(self,
                 batch_size,   # 数据的批次大小
                 input_size,   # 数据的输入层大小
                 hidden_size,  # 数据的隐藏层大小
                 output_size,  # 数据的输出层大小
                 poses):       # 数据种类
        super(Auto_Point_Encoder, self).__init__()

        # 数据层大小
        self.batch_size = batch_size
        # 输入层大小 （pointnet++ 隐藏层）
        self.input_size = input_size
        # 中间层大小
        self.hidden_size = hidden_size
        # 输出层大小（均值、方差）
        self.output_size = output_size
        # 关节数目
        self.poses       = poses

        # 均值、方差编码器
        self.mu_Fitter  = MultiFittingLayer(batch_size, input_size, hidden_size ,output_size)
        self.mu_Conv1D  = MultiConv1DLayer(batch_size, input_size, hidden_size ,output_size)
        self.var_Fitter = MultiFittingLayer(batch_size, input_size, hidden_size ,output_size)
        self.var_Conv1D = MultiConv1DLayer(batch_size, input_size, hidden_size ,output_size)
        self.attention  = Self_Attention(batch_size, hidden_size // 4 - 1, poses)
        self.activation1 = torch.nn.Sigmoid()
        self.activation2 = torch.nn.ReLU()

        # 输入：batch x hidden x hidden x 1
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, 5),
            torch.nn.MaxPool2d(3, 2),
            torch.nn.BatchNorm2d(32),
            torch.nn.Conv2d(32, 128, 5),
            torch.nn.MaxPool2d(3, 2), # hidden_size / 4 * hidden_size / 4
            torch.nn.BatchNorm2d(128),
            torch.nn.Conv2d(128, poses, 5), # hidden_size / 4 * hidden_size / 4
            torch.nn.BatchNorm2d(poses),
        )

    def forward(self, x):
        mu_fitter  = self.mu_Fitter(x)
        var_fitter = self.var_Fitter(x)
        mu_ = self.mu_Conv1D(mu_fitter)
        var_ = self.var_Conv1D(var_fitter)

        # 提供数据
        mu  = mu_.view(-1, 16, mu_.shape[2], mu_.shape[3])
        var = var_.view(-1, 16, var_.shape[2], var_.shape[3])

        eps = torch.randn_like(mu)
        z   = mu + eps * var
        z   = self.model(z)
        z   = self.activation2(z)
        return z

def parse_pose(pose, channel_1, channel_2, img_width=640, img_height=480, origin_pose=11):
    pose_size  = pose.shape[0] # 防止多人
    poses_part = np.array([])

    for jj in range(pose_size):
        pose_data = np.zeros(shape=[channel_1, channel_2], dtype=np.float32)

        xx = pose[jj][0] / img_width * channel_1
        yy = pose[jj][1] / img_height * channel_2

        if xx < 1e-6 and yy < 1e-6:
            if poses_part.size == 0:
                poses_part = pose_data
            else:
                poses_part = np.vstack([poses_part, pose_data])
            continue

        for iii in range(channel_1):
            for jjj in range(channel_2):
                pose_data[iii][jjj] = ((np.exp(-((iii-xx)**2.0+(jjj-yy)**2.0)/2.0/5.0))/
                                           (2.0*np.pi*5.0))
        # 归一化处理
        pose_data = pose_data / np.max(np.max(pose_data))
        if poses_part.size == 0:
            poses_part = pose_data
        else:
            poses_part = np.vstack([poses_part, pose_data])

    poses_part = np.reshape(poses_part, [pose_size, channel_1, channel_2])

    if origin_pose != pose_size:
        new_poses_part = np.zeros(shape=[origin_pose, channel_1, channel_2], dtype=np.float32)
        iter_num = int(new_poses_part.shape[0] / origin_pose)
        for ii in range(iter_num):
            new_poses_part[0] += poses_part[ii * origin_pose + 0]
            new_poses_part[1] += poses_part[ii * origin_pose + 1]
            new_poses_part[2] += poses_part[ii * origin_pose + 2]
            new_poses_part[3] += poses_part[ii * origin_pose + 3]
            new_poses_part[4] += poses_part[ii * origin_pose + 4]
            new_poses_part[5] += poses_part[ii * origin_pose + 5]
            new_poses_part[6] += poses_part[ii * origin_pose + 6]
            new_poses_part[7] += poses_part[ii * origin_pose + 7]
            new_poses_part[8] += poses_part[ii * origin_pose + 8]
            new_poses_part[9] += poses_part[ii * origin_pose + 9]
            new_poses_part[10] += poses_part[ii * origin_pose + 10]
        for ii in range(origin_pose):
            if np.max(np.max(new_poses_part[ii])) > 1e-5:
                new_poses_part[ii] /= np.max(np.max(new_poses_part[ii]))

        poses = new_poses_part
        return poses * 100.0

    poses = poses_part
    return poses * 100.0

def get_mse_loss_function():
    return torch.nn.MSELoss() # 分布外形 最直接的方法

def get_bce_loss_function():
    return torch.nn.BCELoss() # 对比分布相似程度

def get_kl_loss_function():
    return torch.nn.KLDivLoss() # compare KL loss

def mse_loss_(pred, label):
    return torch.sum(torch.abs(pred - label))

def bce_loss_(pred, label):
    return torch.sum(-(1 - label) * torch.log(1 - pred + 1e-6) - label * torch.log(pred + 1e-6))

# coder = Auto_Point_Encoder(32, 1024, 1024, -1, 11).cuda()
# data  = torch.rand([32, 1024]).cuda()
# coder.forward(data)