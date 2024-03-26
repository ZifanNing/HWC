# -*- coding: utf-8 -*-

'''
------------------------------------------------------------------------------

Copyright (C) 2019 Université catholique de Louvain (UCLouvain), Belgium.

Licensed under the Apache License, Version 2.0 (the 'License');
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an 'AS IS' BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

------------------------------------------------------------------------------

 'module.py' - Definition of hooks that allow performing FA, DFA, and DRTP training.

 Project: DRTP - Direct Random Target Projection

 Authors:  C. Frenkel and M. Lefebvre, Université catholique de Louvain (UCLouvain), 09/2019

 Cite/paper: C. Frenkel, M. Lefebvre and D. Bol, 'Learning without feedback: Direct random target projection
    as a feedback-alignment algorithm with layerwise feedforward training,' arXiv preprint arXiv:1909.01311, 2019.

------------------------------------------------------------------------------
'''

import torch
import torch.nn as nn
from models.rp.function import trainingHook
import numpy as np
import random
from scipy.stats import beta

class TrainingHook(nn.Module):
    def __init__(self, label_features, dim_hook, train_mode):
        super(TrainingHook, self).__init__()
        self.train_mode = train_mode
        assert train_mode in ['BP', 'FA', 'DFA', 'DRTP', 'sDFA', 'shallow'], '=== ERROR: Unsupported hook training mode ' + train_mode + '.'

        # Feedback weights definition (FA feedback weights are handled in the FA_wrapper class)
        if self.train_mode in ['DFA', 'DRTP', 'sDFA']:
            self.fixed_fb_weights = nn.Parameter(torch.Tensor(torch.Size(dim_hook)))
            self.reset_weights()
        else:
            self.fixed_fb_weights = None

    def reset_weights(self):
        model = 'masked' # gl.get_value('args').fixedB
        # model = '0~1' #kaiming, 0~1, -1~1, eye,
        # model = '-1~1'  # kaiming, 0~1, -1~1, eye,
        # model = 'triu'  # kaiming, 0~1, -1~1, eye,
        # model = 'tril'
        # model = '0~1'
        # model = 'noeye'

        if model == 'masked':
            random.seed(3154)
            num_feature = self.fixed_fb_weights.shape[1]
            num_class = self.fixed_fb_weights.shape[0]
            a = list(range(num_feature))
            b = None
            for _ in range(num_class):
                c = random.sample(a, int(num_feature / num_class))
                c.sort()
                a = list(set(a).difference(set(c)))
                if b is None:
                    b = torch.zeros((1, num_feature))
                    b[0, c] = 1
                else:
                    b_add = torch.zeros((1, num_feature))
                    b_add[0, c] = 1
                    b = torch.cat((b, b_add), dim=0)
            y = torch.rand(self.fixed_fb_weights.data.shape)
            y = 2 * y - 1
            self.fixed_fb_weights = nn.Parameter(y)
            # self.fixed_fb_weights = nn.Parameter(nn.init.orthogonal_(y) * b)

        if model == 'kaiming':
            torch.nn.init.kaiming_uniform_(self.fixed_fb_weights)
            print(self.fixed_fb_weights.max(), self.fixed_fb_weights.min())
            print(model)
        if model == '0~1':
            if len(self.fixed_fb_weights.shape) == 2:
                x = torch.rand(self.fixed_fb_weights.data.shape)
                y = torch.eye(x.shape[-2], x.shape[-1])
                self.fixed_fb_weights = nn.Parameter(x * y)
            if len(self.fixed_fb_weights.shape) == 4:
                x = torch.ones(self.fixed_fb_weights.data.shape[:3])
                y = torch.diag_embed(x)
                z = torch.rand(self.fixed_fb_weights.data.shape)
                self.fixed_fb_weights = nn.Parameter(z * y)
            print(model)
        if model == '-1~1':
            if len(self.fixed_fb_weights.shape) == 2:
                x = torch.rand(self.fixed_fb_weights.data.shape) * 2 - 1
                y = torch.eye(x.shape[-2], x.shape[-1])
                self.fixed_fb_weights = nn.Parameter(x * y)
            if len(self.fixed_fb_weights.shape) == 4:
                x = torch.ones(self.fixed_fb_weights.data.shape[:3])
                y = torch.diag_embed(x)
                z = torch.rand(self.fixed_fb_weights.data.shape) * 2 - 1
                self.fixed_fb_weights = nn.Parameter(z * y)
            print(model)
        if model == 'triu': #上三角
            x = np.ones(self.fixed_fb_weights.shape)
            x = np.triu(x)
            self.fixed_fb_weights = nn.Parameter(torch.from_numpy(x).float())
            print(model)
        if model == 'tril': #下三角
            x = np.ones(self.fixed_fb_weights.shape)
            x = np.tril(x)
            self.fixed_fb_weights = nn.Parameter(torch.from_numpy(x).float())
            print(model)
        if model == 'noeye':
            if len(self.fixed_fb_weights.shape) == 4:
                x = torch.rand(self.fixed_fb_weights.data.shape) * 2 - 1
                x = np.triu(x.cpu())
                x = torch.from_numpy(x).cuda()
                x = x + x.permute(0, 1, 3, 2)
                y = torch.ones(x.shape[:3])
                y = 1 - torch.diag_embed(y)
                self.fixed_fb_weights = nn.Parameter((x * y).float())
            if len(self.fixed_fb_weights.shape) == 2:
                x = torch.rand(self.fixed_fb_weights.data.shape) * 2 - 1
                x = np.triu(x.cpu())
                # x = torch.from_numpy(x).cuda()
                z = np.zeros((x.shape[0],x.shape[1]-x.shape[0]))
                z = np.c_[x.T[:x.shape[0],:],z]
                x = torch.from_numpy(x + z).cuda()
                y = 1- torch.eye(x.shape[0], x.shape[1])
                self.fixed_fb_weights = nn.Parameter((x * y).float())
            print(model)
        if model == '1': #单位矩阵
            if len(self.fixed_fb_weights.shape) == 2:
                y = torch.eye(self.fixed_fb_weights.data.shape[-2], self.fixed_fb_weights.data.shape[-1])
                self.fixed_fb_weights = nn.Parameter(y)
            if len(self.fixed_fb_weights.shape) == 4:
                x = torch.ones(self.fixed_fb_weights.data.shape[:3])
                y = torch.diag_embed(x)
                self.fixed_fb_weights = nn.Parameter(y)
            print(model)
        if model == '11N': #正态分布-1~1
            y = torch.randn(self.fixed_fb_weights.data.shape)
            y = y / abs(y).max()
            self.fixed_fb_weights = nn.Parameter(y)
            print(model)
        if model == '11P': #泊松分布-1~1
            y = np.random.poisson(lam=5, size=self.fixed_fb_weights.data.shape)
            y = torch.from_numpy(y).cuda()
            y = torch.true_divide(y, abs(y.max()))
            y = y * 2 - 1
            self.fixed_fb_weights = nn.Parameter(y)
            print(model)
        if model == '11U': #均匀分布-1~1
            y = torch.rand(self.fixed_fb_weights.data.shape)
            y = 2 * y - 1
            self.fixed_fb_weights = nn.Parameter(y)
            print(model)
        if model == '105U': #均匀分布-1~-0.5
            y = torch.rand(self.fixed_fb_weights.data.shape)
            y = 1 / 2 * y - 1
            self.fixed_fb_weights = nn.Parameter(y)
            print(model)
        if model == '050U': #均匀分布-1~-0.5
            y = torch.rand(self.fixed_fb_weights.data.shape)
            y = 1 / 2 * y - 0.5
            self.fixed_fb_weights = nn.Parameter(y)
            print(model)
        if model == '005U': #均匀分布-1~-0.5
            y = torch.rand(self.fixed_fb_weights.data.shape)
            y = 1 / 2 * y
            self.fixed_fb_weights = nn.Parameter(y)
            print(model)
        if model == '051U': #均匀分布-1~-0.5
            y = torch.rand(self.fixed_fb_weights.data.shape)
            y = 1 / 2 * y + 0.5
            self.fixed_fb_weights = nn.Parameter(y)
            print(model)
        if model == 'EyeU': #单位矩阵
            if len(self.fixed_fb_weights.shape) == 2:
                y = torch.rand(self.fixed_fb_weights.data.shape)
                y = 2 * y - 1
                z = torch.eye(self.fixed_fb_weights.data.shape[-2], self.fixed_fb_weights.data.shape[-1])
                self.fixed_fb_weights = nn.Parameter(y * z)
            if len(self.fixed_fb_weights.shape) == 4:
                x = torch.rand(self.fixed_fb_weights.data.shape[:3]) * 2 - 1
                y = torch.diag_embed(x)
                self.fixed_fb_weights = nn.Parameter(y)
            print(model)
        if model == 'TriuU': #上三角
            x = torch.rand(self.fixed_fb_weights.data.shape) * 2 - 1
            x = np.triu(x.cpu())
            self.fixed_fb_weights = nn.Parameter(torch.from_numpy(x).float().cuda())
            print(model)
        if model == 'TrilU': #下三角
            x = torch.rand(self.fixed_fb_weights.data.shape) * 2 - 1
            x = np.tril(x.cpu())
            self.fixed_fb_weights = nn.Parameter(torch.from_numpy(x).float().cuda())
            print(model)

        if 'Rank' in model: #下三角
            rank = int(float(model.split('_')[1]) * self.fixed_fb_weights.shape[-2])
            if len(self.fixed_fb_weights.shape) == 4:
                x = torch.rand(self.fixed_fb_weights.data.shape) * 2 - 1
                y = torch.ones((self.fixed_fb_weights.shape[0], self.fixed_fb_weights.shape[1], rank, self.fixed_fb_weights.shape[3]))
                z = torch.zeros((self.fixed_fb_weights.shape[0], self.fixed_fb_weights.shape[1],self.fixed_fb_weights.shape[2] - rank, self.fixed_fb_weights.shape[3]))
                y = torch.cat((y, z), dim=2)
                x = x * y
                k = rank
                while k < self.fixed_fb_weights.shape[2]:
                    x[:, :, k, :] = x[:, :, k - rank, :]
                    k += 1
                a = np.arange(self.fixed_fb_weights.shape[2])
                np.random.shuffle(a)
                x = x[:, :, a, :]
                self.fixed_fb_weights = nn.Parameter(x.float().cuda())
            if len(self.fixed_fb_weights.shape) == 2:
                x = torch.rand(self.fixed_fb_weights.data.shape) * 2 - 1
                y = torch.ones((rank, self.fixed_fb_weights.shape[1]))
                z = torch.zeros((self.fixed_fb_weights.shape[0] - rank, self.fixed_fb_weights.shape[1]))
                y = torch.cat((y, z), dim=0)
                x = x * y
                k = rank
                while k < self.fixed_fb_weights.shape[0]:
                    x[k, :] = x[k - rank, :]
                    k += 1
                a = np.arange(self.fixed_fb_weights.shape[0])
                np.random.shuffle(a)
                x = x[a, :]
                self.fixed_fb_weights = nn.Parameter(x.float().cuda())
            print(model)
        if 'Uniform' in model: #均匀分布-1~-0.5
            alpha = float(model.split('_')[1])
            y = torch.rand(self.fixed_fb_weights.data.shape)
            y = 2 * y - 1
            y = alpha * y
            self.fixed_fb_weights = nn.Parameter(y)
            print(model)
        if 'Beta' in model: #Beta分布
            alpha_value = float(model.split('_')[1])
            bata_value = float(model.split('_')[2])
            alpha = float(model.split('_')[3])
            y = torch.rand(self.fixed_fb_weights.data.shape)
            y = beta(alpha_value, bata_value).pdf(y.cpu())
            y = y / y.max()
            y = 2 * y - 1
            y = alpha * y
            self.fixed_fb_weights = nn.Parameter(torch.from_numpy(y).float().cuda())
            print(model)
        if 'EXP' in model: #指数分布
            lambd = float(model.split('_')[1])
            y = torch.rand(self.fixed_fb_weights.data.shape)
            y = lambd * np.exp(-lambd * y.cpu())
            y = y - (y.max()+y.min())/2
            self.fixed_fb_weights = nn.Parameter(y.cuda())
            print(model)

        self.fixed_fb_weights.requires_grad = False

    def forward(self, input, labels, y):
        return trainingHook(input, labels, y, self.fixed_fb_weights, self.train_mode if (self.train_mode != 'FA') else 'BP') #FA is handled in FA_wrapper, not in TrainingHook

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.train_mode + ')'
