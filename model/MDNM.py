
import torch
import torch.nn as nn
import torch.nn.init as init


class SynapticLayer(nn.Module):
    def __init__(self, input_dim, dendriteNum, k):
        super().__init__()
        self.k = k
        self.weight = nn.Parameter(torch.Tensor(input_dim, dendriteNum))
        init.xavier_uniform_(self.weight)
        self.theta = nn.Parameter(torch.Tensor(input_dim, dendriteNum))
        init.xavier_uniform_(self.theta)
    def forward(self, x):
        # print("x: ", x.size())
        # print("self.weight: ", self.weight.size())
        # print("self.theta: ", self.theta.size())
        return torch.sigmoid(self.k * (x * self.weight  - self.theta))


class DendriteLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.prod(x, dim = 1)
    

class MembaneLayer(nn.Module):
    def __init__(self, dendriteNum):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(dendriteNum))
        nn.init.uniform_(self.weight, a=-0.1, b=0.1)
    def forward(self, x):
        return torch.sum(self.weight * x, dim=1)


class SomaLayer(nn.Module):
    def __init__(self, ks, thetas):
        super().__init__()
        self.ks = ks
        self.thetas = thetas

    def forward(self, x):
        return torch.sigmoid(self.ks * (x - self.thetas))


class DNMModel(nn.Module):
    def __init__(self, input_dim=18, dendriteNum=7, k=5, ks=0.5, thetas=0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, dendriteNum)
        # 加了mlp的dnm
        self.synapticLayer = SynapticLayer(dendriteNum, dendriteNum, k)
        self.dendriteLayer = DendriteLayer()
        self.membraneLayer = MembaneLayer(dendriteNum)
        self.somaLayer = SomaLayer(ks, thetas)
        self.dendriteNum = dendriteNum

        self.sy2 = SynapticLayer(dendriteNum, dendriteNum, k)
        self.den2 = DendriteLayer()

        self.sy3 = SynapticLayer(dendriteNum, dendriteNum, k)
        self.den3 = DendriteLayer()

        self.sy4 = SynapticLayer(dendriteNum, dendriteNum, k)
        self.den4 = DendriteLayer()

        self.sy5 = SynapticLayer(dendriteNum, dendriteNum, k)
        self.den5 = DendriteLayer()
        self.k = k
        self.ks = ks
        self.thetas = thetas
    def forward(self, x):
        # 加一层mlp
        x = self.fc1(x)

        # 第一层累计残差块
        x_unsqueeze_1_dim = x.unsqueeze(2)
        x_expand_1_dim = x_unsqueeze_1_dim.expand(-1, -1, self.dendriteNum)
        x_synaptic_1_dim = self.synapticLayer(x_expand_1_dim) 
        # 不加残差
        x_dendrite_1_dim = self.dendriteLayer(x_synaptic_1_dim)
        # 累加残差
        # x_dendrite_1_dim = x + self.dendriteLayer(x_synaptic_1_dim)
        
        # 第二层累积残差块
        # x_unsqueeze_2_dim = x_dendrite_1_dim.unsqueeze(2)
        # x_expand_2_dim = x_unsqueeze_2_dim.expand(-1, -1, self.dendriteNum)
        # x_synaptic_2_dim = self.sy2(x_expand_2_dim)
        # x_dendrite_2_dim = x_dendrite_1_dim + self.den2(x_synaptic_2_dim)
        
        # 第三层累计残差块
        # x_unsqueeze_3_dim = x_dendrite_2_dim.unsqueeze(2)
        # x_expand_3_dim = x_unsqueeze_3_dim.expand(-1, -1, self.dendriteNum)
        # x_synaptic_3_dim = self.sy3(x_expand_3_dim)
        # x_dendrite_3_dim = x_dendrite_2_dim + self.den3(x_synaptic_3_dim)
        
        # 第四层累计残差块
        # x_unsqueeze_4_dim = x_dendrite_3_dim.unsqueeze(2)
        # x_expand_4_dim = x_unsqueeze_4_dim.expand(-1, -1, self.dendriteNum)
        # x_synaptic_4_dim = self.sy4(x_expand_4_dim)
        # x_dendrite_4_dim = x_dendrite_3_dim + self.den4(x_synaptic_4_dim)

        # 第五层累计残差块
        # x_unsqueeze_5_dim = x_dendrite_4_dim.unsqueeze(2)
        # x_expand_5_dim = x_unsqueeze_5_dim.expand(-1, -1, self.dendriteNum)
        # x_synaptic_5_dim = self.sy5(x_expand_5_dim)
        # x_dendrite_5_dim = x_dendrite_4_dim + self.den5(x_synaptic_5_dim)

        x_membrane_2_dim = self.membraneLayer(x_dendrite_1_dim)
        x_soma_dim = self.somaLayer(x_membrane_2_dim)

        return x_soma_dim


    


