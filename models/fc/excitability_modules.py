import math
import numpy as np
from numpy import *
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from models.rp.module import TrainingHook

thresh, lens, decay, stp_decay = (0.5, 0.2, 0.2, 0.92)

class ActFun(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, orthogonal_projector_in, orthogonal_projector_out, trace, op, na_mask):
        ctx.save_for_backward(input, orthogonal_projector_in, orthogonal_projector_out, trace, op, na_mask)
        return input.gt(thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, orthogonal_projector_in, orthogonal_projector_out, trace, op, na_mask = ctx.saved_tensors
        grad_input = grad_output.clone()
        # print(grad_input.mean())
        # temp = input > (thresh - lens) # new
        temp = abs(input - thresh) < lens # old
        if trace is not None:
            if op == 0:
                grad_input = torch.mm(grad_input, torch.mm(trace, torch.mm(orthogonal_projector_in, trace.t()))) * 1e-4
                # print(grad_input.mean())
                # grad_input = torch.mm(grad_input, F.interpolate(orthogonal_projector_in, size=grad_input.shape[1]))
            elif op == 1:
                grad_input = torch.mm(grad_input, orthogonal_projector_out)

        if na_mask is not None:
            grad_input = grad_input * na_mask

        # print(grad_input.mean())
        return grad_input * temp.float(), None, None, None, None, None

    # @staticmethod
    # def backward(ctx, grad_h):
    #     z = ctx.saved_tensors
    #     s = torch.sigmoid(z[0])
    #     d_input = (1 - s) * s * grad_h
    #     return d_input

act_fun = ActFun.apply

def linearExcitability(input, weight, excitability=None, bias=None):
    '''
    Applies a linear transformation to the incoming data: :math:`y = c(xA^T) + b`.

    Shape:
        - input:        :math:`(N, *, in\_features)`
        - weight:       :math:`(out\_features, in\_features)`
        - excitability: :math:`(out\_features)`
        - bias:         :math:`(out\_features)`
        - output:       :math:`(N, *, out\_features)`
    (NOTE: `*` means any number of additional dimensions)
    '''

    if type(input) == list:
        spikes = 0
        for i in range(len(input)):
            spikes += input[i]
        input = spikes / len(input)

    if excitability is not None:
        output = input.matmul(weight.t()) * excitability
    else:
        output = input.matmul(weight.t())
    if bias is not None:
        output += bias
    return output

class STDP_masking(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, stdp_mask):
        ctx.save_for_backward(stdp_mask)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        stdp_mask, = ctx.saved_tensors
        grad_input = grad_output.clone()
        # if grad_output.shape[0] == 10:
        #     for i in range(10):
        #         print(grad_input[i, :].mean())
        return grad_input * stdp_mask if stdp_mask is not None else grad_input, None

stdp_masking = STDP_masking.apply

def sort_greater(input, proportion):
    a, b = input.shape
    input_ = input.view(1, -1)
    _, indices = torch.sort(input_, descending=True)
    num = int(a * b * (1 - proportion))
    input_[indices < num] = 1
    input_[indices >= num] = 0
    output = input_.view(a, b)

    return output

def get_using_label(labels):
    min_label = int(labels.min().item())
    max_label = int(labels.max().item())
    label_list = list(range(min_label, max_label + 1))
    for label in label_list:
        index = (labels == label)
        if len(index) == 0:
            label_list.remove(label)

    return label_list, min_label, max_label

def integrate_stdp(stdp_list, mark):
    '''integrate stdp matrix by label(list) or task(int) '''

    if type(mark) is list:
        max_label = max(mark)
        if len(stdp_list) < (max_label + 1):
            for i in range(max_label + 1 - len(stdp_list)):
                stdp_list.append(torch.zeros_like(stdp_list[0]))

        stdp = torch.zeros_like(stdp_list[0])
        for i in range(len(stdp_list)):
            if i not in mark:
                stdp[stdp < stdp_list[i]] = stdp_list[i][stdp < stdp_list[i]]
    else:
        if len(stdp_list) < (mark + 1):
            for i in range(mark + 1 - len(stdp_list)):
                stdp_list.append(torch.zeros_like(stdp_list[0]))

        stdp = torch.zeros_like(stdp_list[0])
        for i in range(len(stdp_list)):
            if i != mark:
                stdp[stdp < stdp_list[i]] = stdp_list[i][stdp < stdp_list[i]]

    return stdp

def update_p(x, p, schedule):
    with torch.no_grad():
        intermediate = torch.mm(p, x.unsqueeze(1))
        delta_p = torch.mm(intermediate, intermediate.t()) / (0.9 * 0.001 ** schedule + torch.mm(x.unsqueeze(0), intermediate))
        new_p = p - delta_p # * 10

    return new_p

def spikinglinearExcitability(input, weight, excitability=None, bias=None, stdp_mask=None):
    '''
    Applies a linear transformation to the incoming data: :math:`y = c(xA^T) + b`.

    Shape:
        - input:        :math:`(N, *, in\_features)`
        - weight:       :math:`(out\_features, in\_features)`
        - excitability: :math:`(out\_features)`
        - bias:         :math:`(out\_features)`
        - output:       :math:`(N, *, out\_features)`
    (NOTE: `*` means any number of additional dimensions)
    '''

    # if type(input) == list:
    #     spikes = 0
    #     for i in range(len(input)):
    #         spikes += input[i]
    #     input = spikes / len(input)

    n_weight = stdp_masking(weight, stdp_mask.detach()) if stdp_mask is not None else weight

    if excitability is not None:
        output = input.matmul(n_weight.t()) * excitability
    else:
        output = input.matmul(n_weight.t())

    return output

class LinearExcitability(nn.Module):
    '''Applies a linear transformation to the incoming data: :math:`y = c(Ax) + b`

    Args:
        in_features:    size of each input sample
        out_features:   size of each output sample
        bias:           if 'False', layer will not learn an additive bias-parameter (DEFAULT=True)
        excitability:   if 'False', layer will not learn a multiplicative excitability-parameter (DEFAULT=True)

    Shape:
        - input:    :math:`(N, *, in\_features)` where `*` means any number of additional dimensions
        - output:   :math:`(N, *, out\_features)` where all but the last dimension are the same shape as the input.

    Attributes:
        weight:         the learnable weights of the module of shape (out_features x in_features)
        excitability:   the learnable multiplication terms (out_features)
        bias:           the learnable bias of the module of shape (out_features)
        excit_buffer:   fixed multiplication variable (out_features)

    Examples::

        >>> m = LinearExcitability(20, 30)
        >>> input = autograd.Variable(torch.randn(128, 20))
        >>> output = m(input)
        >>> print(output.size())
    '''

    def __init__(self, in_features, out_features, bias=True, excitability=False, excit_buffer=False):
        super(LinearExcitability, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if excitability:
            self.excitability = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('excitability', None)
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        if excit_buffer:
            buffer = torch.Tensor(out_features).uniform_(1,1)
            self.register_buffer('excit_buffer', buffer)
        else:
            self.register_buffer('excit_buffer', None)
        self.reset_parameters()

    def reset_parameters(self):
        '''Modifies the parameters 'in-place' to reset them at appropriate initialization values'''
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        nn.init.orthogonal_(self.weight)
        if self.excitability is not None:
            self.excitability.data.uniform_(1, 1)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        '''Running this model's forward step requires/returns:
        INPUT: -[input]: [batch_size]x[...]x[in_features]
        OUTPUT: -[output]: [batch_size]x[...]x[hidden_features]'''
        if self.excit_buffer is None:
            excitability = self.excitability
        elif self.excitability is None:
            excitability = self.excit_buffer
        else:
            excitability = self.excitability * self.excit_buffer
        return linearExcitability(input, self.weight, excitability, self.bias)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'in_features=' + str(self.in_features) + ', out_features=' + str(self.out_features) + ')'

class SpikingLinearExcitability(nn.Module):
    '''Applies a spiking version of linear transformation to the incoming data: :math:`y = c(Ax) + b`'''

    def __init__(self, args, in_features, out_features, bias=True, excitability=False, excit_buffer=False, membrane_reserve=False):
        super(SpikingLinearExcitability, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.last_layer = True if self.out_features == args.classes else False
        self.time_window = args.time_window
        self.membrane_reserve = membrane_reserve
        self.membrane_potential = None
        self.na_update_rate = args.na_update_rate
        self.stdp_update_rate = args.stdp_update_rate
        self.current_multiple = args.current_multiple
        self.current_bias = args.current_bias

        self.use_op = args.use_op
        self.op = args.op
        self.orthogonal_projector_in = torch.eye(in_features) # * 1e-3
        # print('trace of orthogonal_projector_in:', linalg.matrix_rank(self.orthogonal_projector_in.cpu().numpy()))
        self.orthogonal_projector_out = torch.eye(out_features)  # * 1e-3
        # print('trace of orthogonal_projector_out', linalg.matrix_rank(self.orthogonal_projector_out.cpu().numpy()))

        self.use_stp = args.use_stp
        self.stp = None 
        self.stp_decay = 0.2 # Parameter(torch.rand(self.out_features) )
        self.stp_strength = torch.ones(self.out_features, self.in_features)  * 0

        self.scenario = args.scenario
        self.new_task = list(range(args.tasks))
        self.trained_task = []
        self.na_list = [torch.zeros(self.out_features) ]
        self.stdp_list = [torch.zeros((self.out_features, self.in_features)) ]
        self.recorded_x = torch.zeros((self.in_features)) 
        self.recorded_spike = torch.zeros((self.out_features)) 
        self.new_learning_phase = False
        self.schedule = 0

        self.spike = None
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.hook = TrainingHook(label_features=None, dim_hook=(10, out_features), train_mode='DRTP')
        if excitability:
            self.excitability = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('excitability', None)
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        if excit_buffer:
            buffer = torch.Tensor(out_features).uniform_(1,1)
            self.register_buffer('excit_buffer', buffer)
        else:
            self.register_buffer('excit_buffer', None)

        self.mask_mode = args.mask_mode
        self.masking = args.masking
        self.hard_masking = args.hard_masking
        print('hard_masking:', self.hard_masking)
        self.na_threshold = args.na_threshold # if not self.last_layer else 0.01
        self.stdp_threshold = args.stdp_threshold
        self.calculate_with_firerate = args.calculate_with_firerate

        self.reset_parameters()

    def reset_parameters(self):
        '''Modifies the parameters 'in-place' to reset them at appropriate initialization values'''
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.excitability is not None:
            self.excitability.data.uniform_(1, 1)

    def forward(self, input, labels, task=None):
        '''Running this model's forward step requires/returns:
        INPUT: -[input]: [batch_size]x[...]x[in_features]
        OUTPUT: -[output]: [batch_size]x[...]x[hidden_features]'''

        # if task is not None:
        #     print('task :', task)
        if type(input) == list:
            batch_size = input[0].shape[0]
        else:
            batch_size = input.shape[0]

        if labels is not None:
            label_list, _, _ = get_using_label(labels)

        self.sum_x = 0
        self.sum_spike = 0
        sum_x = 0
        sum_spike = 0
        outputs = []
        if self.membrane_reserve:
            membrane_potential = self.membrane_potential
            stp = self.stp
        else:
            membrane_potential = None
            stp = None

        if self.excit_buffer is None:
            excitability = self.excitability
        elif self.excitability is None:
            excitability = self.excit_buffer
        else:
            excitability = self.excitability * self.excit_buffer

        with torch.no_grad():
            stp_strength = self.stp_strength

            if task in self.new_task:
                self.trained_task.append(task)

            if labels is not None:
                if self.mask_mode == 'task':
                    na_mask_temp = integrate_stdp(self.na_list, task)
                    index_na = (self.na_list[task] - na_mask_temp) > 0.95

                    stdp_mask_temp = integrate_stdp(self.stdp_list, task)
                    index_stdp = (self.stdp_list[task] - stdp_mask_temp) > 0.95
                elif self.mask_mode == 'label':
                    na_mask_temp = integrate_stdp(self.na_list, label_list)
                    temp_for_index_na = torch.zeros_like(self.na_list[0])
                    for label in label_list:
                        temp_for_index_na[temp_for_index_na < self.na_list[label]] = self.na_list[label][temp_for_index_na < self.na_list[label]]
                    index_na = (temp_for_index_na - na_mask_temp) > 0.95

                    stdp_mask_temp = integrate_stdp(self.stdp_list, label_list)
                    temp_for_index_stdp = torch.zeros_like(self.stdp_list[0])
                    for label in label_list:
                        temp_for_index_stdp[temp_for_index_stdp < self.stdp_list[label]] = self.stdp_list[label][temp_for_index_stdp < self.stdp_list[label]]
                    index_stdp = (temp_for_index_stdp - stdp_mask_temp) > 0.95

                if self.hard_masking:
                    if na_mask_temp[na_mask_temp >= self.na_threshold].shape[0] > 0:
                        na_mask_temp[na_mask_temp >= self.na_threshold] = 1
                    if na_mask_temp[na_mask_temp < self.na_threshold].shape[0] > 0:
                        na_mask_temp[na_mask_temp < self.na_threshold] = 0

                    if stdp_mask_temp[stdp_mask_temp >= self.stdp_threshold].shape[0] > 0:
                        stdp_mask_temp[stdp_mask_temp >= self.stdp_threshold] = 1
                    if stdp_mask_temp[stdp_mask_temp < self.stdp_threshold].shape[0] > 0:
                        stdp_mask_temp[stdp_mask_temp < self.stdp_threshold] = 0

                if index_na.sum() != 0:
                    na_mask_temp[index_na] = 0

                if index_stdp.sum() != 0:
                    stdp_mask_temp[index_stdp] = 0

                na_mask = 1 - na_mask_temp
                stdp_mask = 1 - stdp_mask_temp
            else:
                na_mask = None
                stdp_mask = None

            if not self.masking:
                na_mask = None
                stdp_mask = None

            if labels is not None:
                trace = torch.zeros_like(self.stdp_list[0])
                if self.mask_mode == 'label':
                    for label in label_list:
                        # trace[trace < self.stdp_list[label]] = self.stdp_list[label][trace < self.stdp_list[label]]
                        trace += self.stdp_list[label]
                    trace /= len(label_list)
                elif self.mask_mode == 'task':
                    trace = self.stdp_list[task]
            else:
                trace = None

            if task in self.new_task:
                self.new_task.remove(task)
                if self.scenario in ['task', 'class']:
                    print('Task:', label_list)
                if na_mask is not None:
                    if self.last_layer:
                        print('na:', na_mask[label_list].mean())
                    else:
                        print('na:', na_mask.mean())
                if stdp_mask is not None:
                    if self.last_layer:
                        print('stdp:', stdp_mask[label_list, :].mean())
                    else:
                        print('stdp:', stdp_mask.mean())

        # if stdp_mask is not None:
        #     print(stdp_mask.mean())

        if self.na_threshold == 100.0:
            self.na_mask = None
        if self.stdp_threshold == 100.0:
            self.stdp_mask = None

        for i in range(self.time_window):
            if type(input) == list:
                x = input[i]
            else:
                x = input > torch.rand(input.size()) 
                x = x.float()

            x = x.view(batch_size, -1)

            if stp is None:
                stp = torch.zeros((batch_size, self.out_features)) 

            # stdp_mask = None
            ori_current = spikinglinearExcitability(x, self.weight, excitability, self.bias, stdp_mask)
            current = self.current_multiple * torch.sigmoid(ori_current) - self.current_bias

            if membrane_potential is None:
                membrane_potential = torch.zeros((batch_size, self.out_features))  + current
            else:
                membrane_potential = membrane_potential * decay * (1 - spike) + current

            # na_mask = None
            # print(stp.mean())
            spike = act_fun(membrane_potential + stp, self.orthogonal_projector_in, self.orthogonal_projector_out, trace, (torch.zeros(1) if self.op == 'in' else torch.ones(1)) if self.use_op else None, na_mask) if self.use_op and (not self.last_layer) else act_fun(membrane_potential + stp, None, None, None, None, na_mask)

            if self.use_stp:
                stp = stp * self.stp_decay + x.clone().detach().matmul(stp_strength.t())

            ### na_mask = None-- Updating STDP Masks with frame --##

            with torch.no_grad():
                if (not self.calculate_with_firerate) and (labels is not None):
                    na_all = spike
                    stdp_all = torch.bmm(spike.unsqueeze(2), x.unsqueeze(1))

                    # stp_strength = stp_strength + stdp_all.mean(dim=0) * (1 - stp_strength) * 1e-4 # - (1 - stdp_all.mean(dim=0)) * stp_strength * 1e-4

                    # stp_strength = torch.clamp(stp_strength, 0, 0.0001)

                    if self.mask_mode == 'task':
                        na_all = na_all.mean(dim=0)
                        na_old = self.na_list[task]
                        na_temp = torch.zeros_like(na_old)
                        na_temp = self.na_update_rate * na_all + (1 - self.na_update_rate) * na_old
                        self.na_list[task] = na_temp
                    elif self.mask_mode == 'label':
                        for label in label_list:
                            index = (labels == label)
                            if len(index) != 0:
                                na_label = na_all[index, :].mean(dim=0)
                                na_old = self.na_list[label]
                                na_temp = torch.zeros_like(na_old)
                                na_temp = self.na_update_rate * na_label + (1 - self.na_update_rate) * na_old
                                self.na_list[label] = na_temp

                    if self.mask_mode == 'task':
                        stdp_all = stdp_all.mean(dim=0)
                        stdp_old = self.stdp_list[task]
                        stdp_temp = torch.zeros_like(stdp_old)
                        stdp_temp = self.stdp_update_rate * stdp_all + (1 - self.stdp_update_rate) * stdp_old
                        self.stdp_list[task] = stdp_temp
                    elif self.mask_mode == 'label':
                        for label in label_list:
                            index = (labels == label)
                            if len(index) != 0:
                                stdp_label = stdp_all[index, :].mean(dim=0)
                                stdp_old = self.stdp_list[label]
                                stdp_temp = torch.zeros_like(stdp_old)
                                stdp_temp = self.stdp_update_rate * stdp_label + (1 - self.stdp_update_rate) * stdp_old
                                self.stdp_list[label] = stdp_temp

            if self.calculate_with_firerate:
                sum_x += x
                sum_spike += spike
            self.sum_x += x
            self.sum_spike += spike
            outputs.append(spike)

        with torch.no_grad():
            if self.new_learning_phase:
                old_in = self.orthogonal_projector_in
                old_out = self.orthogonal_projector_out

                self.orthogonal_projector_in = update_p(self.sum_x.mean(dim=0) / self.time_window, self.orthogonal_projector_in, self.schedule)
                self.orthogonal_projector_out = update_p(self.sum_spike.mean(dim=0) / self.time_window, self.orthogonal_projector_out, self.schedule)

                # print('norm_in', torch.norm(self.orthogonal_projector_in))
                self.orthogonal_projector_in = update_p(self.recorded_x, self.orthogonal_projector_in, self.schedule)
                # print('trace of orthogonal_projector_in:', linalg.matrix_rank(self.orthogonal_projector_in.cpu().numpy()))
                # print('norm_out', torch.norm(self.orthogonal_projector_out))
                self.orthogonal_projector_out = update_p(self.recorded_spike, self.orthogonal_projector_out, self.schedule)
                # print('trace of orthogonal_projector_out', linalg.matrix_rank(self.orthogonal_projector_out.cpu().numpy()))

                # print('delta_in :', (old_in - self.orthogonal_projector_in).mean())
                # print('delta_out :', (old_out - self.orthogonal_projector_out).mean())

                # print('angle_in :', torch.cosine_similarity(old_in.view(1, -1), self.orthogonal_projector_in.view(1, -1)))
                # print('angle_out :', torch.cosine_similarity(old_out.view(1, -1), self.orthogonal_projector_out.view(1, -1)))

                self.recorded_x = torch.zeros((self.in_features)) 
                self.recorded_spike = torch.zeros((self.out_features)) 
                self.new_learning_phase = False
            else:
                self.recorded_x = (self.sum_x.mean(dim=0) / self.time_window + self.recorded_x) / 2
                self.recorded_spike = (self.sum_spike.mean(dim=0) / self.time_window + self.recorded_spike) / 2

        self.stp_strength = stp_strength.clone().detach()
        # print(self.stp_strength.mean())

        # ##-- Updating STDP Masks with firerate --##

        with torch.no_grad():
            if self.calculate_with_firerate and (labels is not None):
                firerate_x = sum_x / self.time_window
                firerate_spike = sum_spike / self.time_window
                na_all = firerate_spike
                stdp_all = torch.bmm(firerate_spike.unsqueeze(2), firerate_x.unsqueeze(1))

                if self.mask_mode == 'task':
                    na_all = na_all.mean(dim=0)
                    na_old = self.na_list[task]
                    na_temp = torch.zeros_like(na_old)
                    na_temp = self.na_update_rate * na_all + (1 - self.na_update_rate) * na_old
                    if self.last_layer and False:
                        ll_temp = torch.zeros_like(na_mask)
                        ll_temp[label_list, :] = na_temp[label_list, :]
                        self.na_list[task] = ll_temp
                    else:
                        self.na_list[task] = na_temp
                elif self.mask_mode == 'label':
                    for label in label_list:
                        index = (labels == label)
                        if len(index) != 0:
                            na_label = na_all[index, :].mean(dim=0)
                            na_old = self.na_list[label]
                            na_temp = torch.zeros_like(na_old)
                            na_temp = self.na_update_rate * na_label + (1 - self.na_update_rate) * na_old
                            if self.last_layer and False:
                                ll_temp = torch.zeros_like(na_temp)
                                ll_temp[label, :] = na_temp[label, :]
                                self.na_list[label] = ll_temp
                            else:
                                self.na_list[label] = na_temp

                if self.mask_mode == 'task':
                    stdp_all = stdp_all.mean(dim=0)
                    stdp_old = self.stdp_list[task]
                    stdp_temp = torch.zeros_like(stdp_old)
                    stdp_temp = self.stdp_update_rate * stdp_all + (1 - self.stdp_update_rate) * stdp_old
                    if self.last_layer and True:
                        ll_temp = torch.zeros_like(stdp_mask)
                        ll_temp[label_list, :] = stdp_temp[label_list, :]
                        self.stdp_list[task] = ll_temp
                    else:
                        self.stdp_list[task] = stdp_temp
                elif self.mask_mode == 'label':
                    for label in label_list:
                        index = (labels == label)
                        if len(index) != 0:
                            stdp_label = stdp_all[index, :].mean(dim=0)
                            stdp_old = self.stdp_list[label]
                            stdp_temp = torch.zeros_like(stdp_old)
                            stdp_temp = self.stdp_update_rate * stdp_label + (1 - self.stdp_update_rate) * stdp_old
                            if self.last_layer and True:
                                ll_temp = torch.zeros_like(stdp_temp)
                                ll_temp[label, :] = stdp_temp[label, :]
                                self.stdp_list[label] = ll_temp
                            else:
                                self.stdp_list[label] = stdp_temp

        if self.membrane_reserve:
            self.membrane_potential = membrane_potential

        return outputs

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'in_features=' + str(self.in_features) + ', out_features=' + str(self.out_features) + ')'