# Copyright (c) 2021, Parallel Systems Architecture Laboratory (PARSA), EPFL &
# Machine Learning and Optimization Laboratory (MLO), EPFL. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the PARSA, EPFL & MLO, EPFL
#    nor the names of its contributors may be used to endorse or promote
#    products derived from this software without specific prior written
#    permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import torch
import torch.nn.functional as F
from torch.optim import SGD
import numpy as np
import pdb
import itertools as it
import logging
import unittest
import bfp.bfp_utils as tracking

class rounding_modes:
    """
    When converting fp32 tensors to bfp, the rounding mode can be chosen.

    STOC: Stochastic rounding
    DETERM: Deterministic rounding
    """
    STOC, DETERM = 'stoc', 'determ'
    modes = [STOC, DETERM]

def round_tensor(t, mode, device):
    """
    Perform the rounding of the tensor t by using selected mode
    """
    if mode == rounding_modes.STOC:
        if device == "cpu":
            sampled = torch.FloatTensor(t.size(), device = device).uniform_(-0.5, 0.5)
        else:
            #sampled = torch.cuda.FloatTensor(t.size()).uniform_(-0.5, 0.5)
            sampled = (-1) * torch.rand(t.size(), device = device) + 0.5
        return sampled.add_(t).round()
    elif mode == rounding_modes.DETERM:
        return t.round()
    else:
        raise NotImplementedError("Rounding mode %s is not implemented", mode)

def get_exponent(t, epsilon):
    """
    Find the shared exponent of the tensor t.
    The exponent of the largest tensor value is selected as the shared exponent.
    """
    #Exponent is independent of the sign
    t = t.abs()
    #Find the maximum element of the tensor t
    max_v, _ = t.max(dim=1, keepdim=True)
    #Get the exponent of that element (We use ceil because in bfp format, we convert using 0.mantissa_bits instead of fp32's 1.mantissa_bits)
    return (max_v + epsilon).log2().ceil()

def _float_to_bfp(t, mant_bits, epsilon, rounding_mode, device, exp_given=None):
    """
    Convert float tensor t to bfp
    """
    #print(f'...................  {mant_bits}  .................... float to bfp')
    exp = get_exponent(t, epsilon)

    #The interval between two consecutive numbers with that exponent value
    interval = torch.pow(2.0, exp-mant_bits)
    #The maximum representable value with exp
    max_v = torch.pow(2.0, exp) - interval

    # To ensure that we preserve the interval
    t = t/interval
    rounded = round_tensor(t, rounding_mode, device)
    rounded *=  interval

    #To ensure that there is no underflow or overflow
    return torch.min(torch.max(rounded, -max_v), max_v)


# Sparsity scheme 4: Generic any level hierarchial element wise N:M sparsity for BFP/FP32
def sparsity_hierarchial_n_m(t, device, N=[], M=[]):
    assert ((len(N) > 0) and (len(M) > 0) and (len(N) == len(M)))
    t = t.contiguous().view(1, -1)
    for idx in range(len(N)):
        non_zero_idx = torch.nonzero(t, as_tuple=True)
        non_zero_elements = t[non_zero_idx].unsqueeze(0)

        pad_size = M[idx] - (non_zero_elements.shape[1] % M[idx])
        non_zero_elements = F.pad(non_zero_elements, (0, pad_size), 'constant')
        non_zero_elements = non_zero_elements.contiguous().view(-1, M[idx])

        temp_t = torch.abs(non_zero_elements)
        _, sparse_idx = torch.topk(temp_t, k=(M[idx]-N[idx]), dim=1, largest=False)
        zero_mask = torch.full(temp_t.shape, 1).to(device=device)
        zero_mask.scatter_(index=sparse_idx, dim=1, value=0)
        
        non_zero_elements = torch.where(zero_mask==0, 0, non_zero_elements)
        non_zero_elements = non_zero_elements.contiguous().view(1, -1)
        non_zero_elements = non_zero_elements.narrow(-1, 0, (non_zero_elements.shape[1]-pad_size))
        t = torch.scatter(t, 1, non_zero_idx[1].unsqueeze(0), non_zero_elements)
    
    return t

# Sparsity scheme 4: FP32 version
def fp32_sparsity_hierarchial_n_m(t, device, N=[], M=[]):
    assert ((len(N) > 0) and (len(M) > 0) and (len(N) == len(M)))
    orig_shape = t.shape
    sparse_t = sparsity_hierarchial_n_m(t, device, N, M)
    return sparse_t.contiguous().view(orig_shape)

# Sparsity scheme 4: BFP version
def bfp_sparsity_hierarchial_n_m(t, mant_bits, epsilon, rounding_mode, device, N=[], M=[], sgd_update=False, bit_range=[], exp_given=None):
    assert ((len(N) > 0) and (len(M) > 0) and (len(N) == len(M)))
    bfp_t = _float_to_bfp(t, mant_bits, epsilon, rounding_mode, device)
    sparse_bfp_t = sparsity_hierarchial_n_m(bfp_t, device, N, M)
    return sparse_bfp_t


def quantize_and_sparsify(t, mant_bits, epsilon, rounding_mode, device, sgd_update=False, sparsity=False, sparsity_frac=0, N=[], M=[], bit_range=[], cols=0, exp_given=None):
    """
    Convert float tensor t to bfp
    """
    if sparsity == False:
        new_t=  _float_to_bfp(t, mant_bits, epsilon, rounding_mode, device)
        return new_t
    else:
        # return block_sparsity_unstructured(t, mant_bits, epsilon, rounding_mode, device, sparsity_frac, sgd_update, unconstrained, bit_range, exp_given)
        # return block_sparsity_one_each_row(t, mant_bits, epsilon, rounding_mode, device, cols, sgd_update, unconstrained, bit_range, exp_given)
        # return bfp_sparsity_unstructured(t, mant_bits, epsilon, rounding_mode, device, sparsity_frac, sgd_update, unconstrained, bit_range, exp_given)
        return bfp_sparsity_hierarchial_n_m(t, mant_bits, epsilon, rounding_mode, device, N, M, sgd_update, bit_range, exp_given)
        # return inter_intra_bfp_sparsity_n_m(t, mant_bits, epsilon, rounding_mode, device, N, M, sgd_update, unconstrained, bit_range, exp_given)





def float_to_bfp_blocked(t, mant_bits, epsilon, rounding_mode, device, bfp_block_size=0,
                       num_format='', weight_mant_bits=0, in_sparsity=False, w_sparsity=False, grad_sparsity=False, rearrange=False, 
                       sparsity_frac=0, N=[0, 0], M=[0, 0], sparsity_num_format='bfp', identifier='',
                       sgd_update=False, bit_range=[], mant_bits_pow=None, mixed_precision='-1,-1', mixed_layer=1):
    """
    Convert fp32 tensor t to bfp with tiling.
    Used for weights (which are handled in the optimizer)
    """
    
    assert (num_format == 'bfp')
    assert (((sparsity_num_format == 'bfp') and (bfp_block_size > 0)) or (sparsity_num_format == 'fp32'))

    intervals=mixed_precision.split(',')
    #print(f'==================== {tracking.current_epoch}')
    for i in range(int(len(intervals)/2)):
        if (mixed_layer==1) or (int(intervals[int(2*i)])<int(tracking.current_epoch)<=int(intervals[int(2*i)+1])):
            #print(f'................................... {tracking.current_epoch}')
            mant_bits = 7

    if in_sparsity == True and identifier == 'in':
        sparsity = True
    elif w_sparsity == True and identifier == 'w':
        sparsity = True
    elif grad_sparsity == True and identifier == 'grad':
        sparsity = True
    else:
        sparsity = False
    
    if sparsity_num_format == 'fp32':
        if sparsity == False:
            return t
        else:
            # return fp32_sparsity_unstructured(t, device, sparsity_frac)
            return fp32_sparsity_hierarchial_n_m(t, device, N, M)
    else:
        if sgd_update:
            mant_bits = weight_mant_bits

        orig_shape = t.shape
        
        if bfp_block_size == 0:
            return _float_to_bfp(t.view(1, -1), mant_bits, epsilon, rounding_mode, device).view(orig_shape)

        padded_shape = list(orig_shape)

        if orig_shape[-1] % bfp_block_size != 0:
            pad_size = bfp_block_size - (orig_shape[-1] % bfp_block_size)
            t = F.pad(t, (0,pad_size),'constant')
            padded_shape[-1] = orig_shape[-1]+pad_size
        
        t = t.contiguous().view(-1, bfp_block_size)
        t = quantize_and_sparsify(t, mant_bits, epsilon, rounding_mode, device, sgd_update=sgd_update, sparsity=sparsity, sparsity_frac=sparsity_frac, N=N, M=M, bit_range=bit_range)
        t = t.contiguous().view(padded_shape)

        return t.narrow(-1,0,orig_shape[-1])



def _get_op_name(name, epsilon, mant_bits, rounding_mode, **kwargs):
    """
    Returns the operation's name that is performed in BFP format
    """
    return  '%s_BFP_%s_%d' % (name, rounding_mode, mant_bits)

def _gen_bfp_op(op, name, bfp_args):
    """
    Do the 'sandwich'
    With an original op:

    out = op(x, y)
    grad_x, grad_y = op_grad(grad_out)

    To the following:
    x_, y_ = input_op(x, y)
    Where input_op(x, y) -> bfp(x), bfp(y)
    and input_op_grad(grad_x, grad_y) -> bfp(grad_x), bfp(grad_y)

    out_ = op(x_, y_)

    out = output_op(out)
    Where output_op(out) -> bfp(out)
    and output_op_grad(grad_out) -> bfp(grad_out)

    This way we garantee that everything in and out of the forward and backward operations is
    properly converted to bfp
    """

    name = _get_op_name(name, **bfp_args)

    class NewOpIn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, w):
            return (float_to_bfp_blocked(x, **bfp_args, identifier='in'), float_to_bfp_blocked(w, **bfp_args, identifier='w'))
            #return MxM_pre_processing(x, w, transpose, **bfp_args)

        @staticmethod
        def backward(ctx, grad_x, grad_w):
            return (grad_x, grad_w)

    NewOpIn.__name__ = name + '_In'
    new_op_in = NewOpIn.apply

    class NewOpOut(torch.autograd.Function):
        @staticmethod
        def forward(ctx, op_out):
            return op_out

        @staticmethod
        def backward(ctx, op_out_grad):
            return float_to_bfp_blocked(op_out_grad, **bfp_args, identifier='grad')

    NewOpOut.__name__ = name + '_Out'
    new_op_out = NewOpOut.apply

    def new_op(x, w, *args, **kwargs):
        x, w = new_op_in(x, w)
        out = op(x, w, *args, **kwargs)
        return new_op_out(out)

    return new_op


_bfp_ops = {}


def _get_bfp_op(op, name, bfp_args):
    """
    Create the bfp version of the operation op
    This function is called when a bfp layer is defined. See BFPConv2d and BFPLinear below
    """
    op_name = _get_op_name(name, **bfp_args)
    if op_name not in _bfp_ops:
        _bfp_ops[name] = _gen_bfp_op(op, name, bfp_args)

    return _bfp_ops[name]


def unpack_bfp_args(kwargs):
    """
    Set up the bfp arguments
    """
    bfp_args = {}
    bfp_argn = [('num_format', 'bfp'),
                ('sparsity_num_format', 'bfp'),
                ('rounding_mode', 'stoc'),
                ('epsilon', 1e-8),
                ('mant_bits', 7),
                ('bfp_block_size', 64),
                ('weight_mant_bits', 15),
                ('in_sparsity', False),
                ('w_sparsity', True),
                ('grad_sparsity', False),
                ('N', [2]),
                ('M', [4]),
                ('rearrange', False),
                ('sparsity_frac', 0),
                ('device', 'cuda'),
                ('mixed_precision', '298,300'),
                ('mixed_layer', 0)]

    for arg, default in bfp_argn:
        if arg in kwargs:
            bfp_args[arg] = kwargs[arg]
            del kwargs[arg]
        else:
            bfp_args[arg] = default
    return bfp_args


def F_linear_bfp(**kwargs):
    """
    bfp linear function

    To be used in the model where F.linear is called
    """
    bfp_args = unpack_bfp_args(kwargs)
    if bfp_args['num_format'] == 'bfp':
        return _get_bfp_op(F.linear, 'linear', bfp_args)
    else:
        return F.linear

def torch_matmul_bfp(mat1, mat2,**kwargs):
    """
    bfp linear function

    To be used in the model where F.linear is called
    """
    bfp_args = unpack_bfp_args(kwargs)
    if bfp_args["num_format"] == "bfp":
        op = _get_bfp_op(torch.matmul, "bmm", bfp_args)
    else:
        op = torch.bmm
    return op(mat1, mat2)


'''
def torch_bmm_bfp(**kwargs):
    """
    bfp linear function

    To be used in the model where F.linear is called
    """
    bfp_args = unpack_bfp_args(kwargs)
    if bfp_args['num_format'] == 'bfp':
        ##print('in if get op')
        return _get_bfp_op(torch.bmm, 'bmm', bfp_args)
    else:
        return torch.bmm

class BFPbmm(torch.bmm):
    """
    bfp linear layer
    """
    def __init__(self, in_features, out_features, **kwargs):
        self.bfp_args = unpack_bfp_args(kwargs)
        super().__init__(in_features, out_features, bias)
        self.num_format = self.bfp_args['num_format']
        self.bmm_op = _get_bfp_op(torch.bmm, 'bmm', bfp_args)
        ##print('........................... bmm init')

    def forward(self, x,y):
        ##print('........................... bmm forward')
        if self.num_format == 'fp32':
            return torch.bmm(x,y, self.bias)
        elif self.num_format == 'bfp':
            l = self.bmm_op(x,y, None)
        else:
            raise NotImplementedError('NumFormat not implemented')
'''

class BFPConv2d(torch.nn.Conv2d):
    """
    bfp convolutional layer
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, **kwargs):
        self.bfp_args = unpack_bfp_args(kwargs)

        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias)
        self.num_format = self.bfp_args['num_format']
        self.conv_op = _get_bfp_op(F.conv2d, 'Conv2d', self.bfp_args)

    def forward(self, input):
        if self.num_format == 'fp32':
            return F.conv2d(input, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)

        elif self.num_format == 'bfp':
            conv = self.conv_op(input, self.weight, None, self.stride,
                                self.padding, self.dilation, self.groups)
            if self.bias is not None:
                return conv + self.bias
            else:
                return conv

        else:
            raise NotImplementedError('NumFormat not implemented')

class BFPLinear(torch.nn.Linear):
    """
    bfp linear layer
    """
    def __init__(self, in_features, out_features, bias=True, **kwargs):
        self.bfp_args = unpack_bfp_args(kwargs)
        super().__init__(in_features, out_features, bias)
        self.num_format = self.bfp_args['num_format']
        self.mant_bits = self.bfp_args['mant_bits']
        self.w_sparsity = self.bfp_args['w_sparsity']
        self.linear_op = _get_bfp_op(F.linear, 'linear', self.bfp_args)
        ##print('........................... BFPLinear init')

    def forward(self, input):
        if self.num_format == 'fp32':
            #print('........................... BFPLinear fp32 forward')
            return F.linear(input, self.weight, self.bias)
        elif self.num_format == 'bfp':
            #print('........................... BFPLinear bfp forward')
            l = self.linear_op(input, self.weight, None)
            if self.bias is not None:
                return l + self.bias
            else:
                return l

        else:
            raise NotImplementedError('NumFormat not implemented')

#if __name__=='__main__':
#    tensortest()



'''
class TestCases(unittest.TestCase):
    def setUp(self):
        """
        Generate all possible bfp numbers that can be represented with given mantissa bits
        Note that we generate the bfp numbers using 0.mantissa_bits instead of fp32's 1.mantissa_bits)

        The implementation of HBFPRepresentables class and representable_numbers function has been adapted from
        https://github.com/TuringMachinegun/float_visualizer/blob/master/visualizer.py
        """
        class HBFPRepresentables():
            def __init__(self, sign, mantissa, exponent):
                self.sign = -1 if sign == "-" else 1
                self.exponent = exponent
                self.bias = 2**(len(exponent)-1)
                self.mantissa = "0" + mantissa

                self.exp_bits = len(exponent)
                self.mant_bits = len(mantissa)

            def to_float(self):
                mantissa_float = self.sign * int(self.mantissa,2)
                mantissa_float /= float(2**self.mant_bits)
                exponent_float = 2**(int(self.exponent, 2)-self.bias)
                return mantissa_float * exponent_float


        def representable_numbers(mant_bits, exp_bits = 10):
            possible_signs = ["-", "+"]
            possible_exponents = ["".join(str(j) for j in i) for i in it.product([0, 1], repeat=exp_bits)]
            possible_hbfp_mantissas = ["".join(str(j) for j in i) for i in it.product([0, 1], repeat=mant_bits)]
            bfp_representable_numbers = []

            for sign in possible_signs:
                for exponent in possible_exponents:
                    numbers_list = []
                    for mantissa in possible_hbfp_mantissas:
                        number = HBFPRepresentables(sign, mantissa, exponent)
                        numbers_list.append(number.to_float())

                    bfp_representable_numbers.append(numbers_list)

            bfp_representable_numbers = np.array(bfp_representable_numbers)
            return bfp_representable_numbers
        self.bfp = representable_numbers

    def test_float_to_bfp(self):
        """
        Generate random fp32 tensors
        Convert them to bfp
        Check if the converted values are contained in the possible bfp numbers
        """
        dtype = torch.float
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        epsilon = 0
        rounding_mode = 'determ'

        for mant_bits in range(12):
            mant_bits +=1
            bfp_numbers = self.bfp(mant_bits)
            for i in range(10):
                t = torch.randn(10, 10, device=device, dtype=dtype)
                b=_float_to_bfp(t, mant_bits, epsilon, rounding_mode, device)
                for tensor_element in b.flatten().tolist():
                    self.assertIn(tensor_element, bfp_numbers, msg="{} is not representable in bfp with {} mantissa bits".format(tensor_element, mant_bits))
                ###print("...Generated tensor {} \nis representable in bfp with {} mantissa bits as \n{}".format(t, mant_bits, b))

    def test_tiled_and_batched(self):
        """
        Generate random fp32 tensors
        Convert them to bfp by using tiled and batched functions
        Check if the converted values are contained in the possible bfp numbers
        """
        dtype = torch.float
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        epsilon = 0
        rounding_mode = 'determ'
        num_format='bfp'
        matrix_h, matrix_w = 32, 32
        tile_size = 15

        for mant_bits in range(12):
            mant_bits +=1
            bfp_numbers = self.bfp(mant_bits)
            for i in range(10):
                t = torch.randn(matrix_h, matrix_w, device=device, dtype=dtype)

                b=float_to_bfp_tiled(t, mant_bits, epsilon, rounding_mode, device, tile_size , num_format)
                for tensor_element in b.flatten().tolist():
                    self.assertIn(tensor_element, bfp_numbers, msg="{} is not representable in bfp with {} mantissa bits".format(tensor_element, mant_bits))
                ###print("...Generated tensor {} \nis representable in bfp with {} mantissa bits as \n{}".format(t, mant_bits, b))

                b=float_to_bfp_batched(t, mant_bits, epsilon, rounding_mode, device, tile_size , num_format)
                for tensor_element in b.flatten().tolist():
                    self.assertIn(tensor_element, bfp_numbers, msg="{} is not representable in bfp with {} mantissa bits".format(tensor_element, mant_bits))
                ##print("...Generated tensor {} \nis representable in bfp with {} mantissa bits as \n{}".format(t, mant_bits, b))

if __name__ == '__main__':
    unittest.main(verbosity=2)
'''

if __name__ == '__main__':
    element_list = [[ 0.0195,  0.0068,  0.0234, -0.0312, -0.0537, -0.0088,  0.0107, -0.0303,
          0.0400, -0.0371, -0.0273, -0.0098,  0.0400,  0.0352,  0.0400,  0.0332,
          0.0420, -0.0459, -0.0449,  0.0703,  0.0537,  0.0459, -0.0020,  0.0186,
         -0.0537, -0.0586,  0.0010,  0.0244, -0.0254, -0.0127, -0.0352, -0.0107,
         -0.0781,  0.0322,  0.0117,  0.0234,  0.0371,  0.0488, -0.0449, -0.0303,
         -0.0293,  0.0303,  0.0234,  0.0352, -0.0312,  0.0449,  0.0010, -0.0137,
         -0.0234, -0.0078, -0.0254,  0.0273, -0.0029,  0.0234, -0.0508, -0.0381,
          0.0146, -0.0293, -0.0244,  0.0459,  0.0391, -0.0391,  0.0303, -0.0342],
    [ 0.0449,  0.0303,  0.0039, -0.0352, -0.0117,  0.0547, -0.0283,  0.0273,
         -0.0273,  0.0186,  0.0107,  0.0254, -0.0264,  0.0020,  0.0049, -0.0088,
          0.0010, -0.0186,  0.0254,  0.0293,  0.0117, -0.0322,  0.0439,  0.0234,
          0.0430, -0.0039,  0.0332, -0.0195,  0.0459,  0.0371, -0.0400,  0.0400,
          0.0576,  0.0117,  0.0439,  0.0283, -0.0244, -0.0264, -0.0234,  0.0303,
          0.0459, -0.0479,  0.0547,  0.0586, -0.0205, -0.0352, -0.0264,  0.0195,
         -0.0625,  0.0000, -0.0332, -0.0039,  0.0312, -0.0254, -0.0234, -0.0312,
          0.0205, -0.0586, -0.0400,  0.0303, -0.0146, -0.0352, -0.0264, -0.0361]]
    
    a = torch.tensor(element_list)
    b = sparsity_hierarchial_n_m(a,'cpu',[2],[4])