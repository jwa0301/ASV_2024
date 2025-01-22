import random
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter
import math
from jiwer import wer
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC


# Wav2Vec2ForCTC의 attention layer를 freeze
def layer_freeze(w2v2, total_layer,frz_layer) :
    w2v2.dropout = nn.Identity()
    w2v2.lm_head = nn.Identity()
    for para in w2v2.parameters():
        para.requires_grad = False

    for name ,child in w2v2.named_children():
        if name == 'wav2vec2':
            for nam,chil in child.named_children():
                if nam == 'encoder' :
                    for na,chi in chil.named_children():
                        if na == 'layers' :
                            for i in range(total_layer):
                                if frz_layer == -1 :
                                    for para in chi.parameters():
                                        para.requires_grad = True
                                elif frz_layer <= i :
                                    for i,para in enumerate(chi[i].parameters()):
                                        para.requires_grad = True
    return w2v2                                                 



class Model(nn.Module):

    def __init__(self,d_args):

        super(Model, self).__init__()
        '''input : (B,1,F,T)'''
        total_transformer_layers = d_args["total_transformer_layers"]
        n_frz_layers = d_args["n_frz_layers"] 
        version = d_args["version"]

        self.w2v2 = Wav2Vec2ForCTC.from_pretrained(version,num_hidden_layers=total_transformer_layers)
                                                 
        self.w2v2 = layer_freeze(self.w2v2, total_transformer_layers, n_frz_layers)
        
        self.conv2d_1  = nn.Conv2d(1, 32, kernel_size=(5,5), stride=(1,1), padding=(2,2))
        self.conv2d_2  = nn.Conv2d(1, 32, kernel_size=(5,5), stride=(1,1), padding=(2,2))
        
        self.max2d_1 = nn.MaxPool2d(kernel_size =(2,2), stride=(2,2),padding = 0)
        
        self.conv2d_3  = nn.Conv2d(32, 32, kernel_size=(1,1), stride=(1,1))
        self.conv2d_4  = nn.Conv2d(32, 32, kernel_size=(1,1), stride=(1,1))
        
        self.bn2d_1 = nn.BatchNorm2d(32)
        
        self.conv2d_5  = nn.Conv2d(32, 48, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.conv2d_6  = nn.Conv2d(32, 48, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        
        self.max2d_2 = nn.MaxPool2d(kernel_size =(2,2), stride=(2,2),padding = 0)
        self.bn2d_2 = nn.BatchNorm2d(48)
        
        
        self.conv2d_7  = nn.Conv2d(48, 48, kernel_size=(1,1), stride=(1,1))
        self.conv2d_8  = nn.Conv2d(48, 48, kernel_size=(1,1), stride=(1,1))
        
        self.bn2d_3 = nn.BatchNorm2d(48)
        
        self.conv2d_9  = nn.Conv2d(48, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.conv2d_10  = nn.Conv2d(48, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        
        self.max2d_3 = nn.MaxPool2d(kernel_size =(2,2), stride=(2,2),padding = 0)
        
        self.conv2d_11  = nn.Conv2d(64, 64, kernel_size=(1,1), stride=(1,1))
        self.conv2d_12  = nn.Conv2d(64, 64, kernel_size=(1,1), stride=(1,1))
        
        self.bn2d_4 = nn.BatchNorm2d(64)
        
        self.conv2d_13  = nn.Conv2d(64, 32, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.conv2d_14  = nn.Conv2d(64, 32, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        
        self.bn2d_5 = nn.BatchNorm2d(32)
        
        
        self.conv2d_15  = nn.Conv2d(32, 32, kernel_size=(1,1), stride=(1,1))
        self.conv2d_16  = nn.Conv2d(32, 32, kernel_size=(1,1), stride=(1,1))
        
        self.bn2d_6 = nn.BatchNorm2d(32)
        
        
        self.conv2d_17  = nn.Conv2d(32, 32, kernel_size=(1,1), stride=(1,1))
        self.conv2d_18  = nn.Conv2d(32, 32, kernel_size=(1,1), stride=(1,1))
        
        self.max2d_4 = nn.MaxPool2d(kernel_size =(2,2), stride=(2,2),padding = 0)
        
        self.adaptavg = nn.AdaptiveAvgPool2d(1)
        
        self.out_layer = nn.Linear(32, 2)
        
    def forward(self, x, Freq_aug=False):
        
        w2v2_out = self.w2v2(x).logits   
        w2v2_out = torch.transpose(w2v2_out,1,2)    # output : (1280,time)
       # print('w2v2 : ', w2v2_out.shape)

        # conv2d로 들어갈 수 있게 차원 확장
        x = w2v2_out.unsqueeze(dim=1)
        
        x1 = self.conv2d_1(x)
        x2 = self.conv2d_2(x)
        max1 = torch.maximum(x1,x2)
        x = self.max2d_1(max1)
        
        x3 = self.conv2d_3(x)
        x4 = self.conv2d_4(x)
        max2 = torch.maximum(x3,x4)
        x = self.bn2d_1(max2)
        
        x5 = self.conv2d_5(x)
        x6 = self.conv2d_6(x)
        max3 = torch.maximum(x5,x6)
        x = self.max2d_2(max3)
        x = self.bn2d_2(x)
        
        x7 = self.conv2d_7(x)
        x8 = self.conv2d_8(x)
        max4 = torch.maximum(x7,x8)
        x = self.bn2d_3(max4)
        
        x9 = self.conv2d_9(x)
        x10 = self.conv2d_10(x)
        max5 = torch.maximum(x9,x10)
        x = self.max2d_3(max5)
        
        x11 = self.conv2d_11(x)
        x12 = self.conv2d_12(x)
        max6 = torch.maximum(x11,x12)
        x = self.bn2d_4(max6)
        
        x13 = self.conv2d_13(x)
        x14 = self.conv2d_14(x)
        max7 = torch.maximum(x13,x14)
        x = self.bn2d_5(max7)
        
        x15 = self.conv2d_15(x)
        x16 = self.conv2d_16(x)
        max8 = torch.maximum(x15,x16)
        x = self.bn2d_6(max8)
        
        x17 = self.conv2d_17(x)
        x18 = self.conv2d_18(x)
        max9 = torch.maximum(x17,x18)
        x = self.max2d_4(max9)
        
        x = self.adaptavg(x)
        last_hidden = x.view(x.size(0),-1)
        output = self.out_layer(last_hidden)
        
        return last_hidden, output
    
    def get_nb_trainable_parameters(self):# -> tuple[int, int]:
        r"""
        Returns the number of trainable parameters and the number of all parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            num_params = param.numel()
            # if using DS Zero 3 and the weights are initialized empty
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel

            # Due to the design of 4bit linear layers from bitsandbytes
            # one needs to multiply the number of parameters by 2 to get
            # the correct number of parameters
            if param.__class__.__name__ == "Params4bit":
                num_params = num_params * 2

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params

        return trainable_params, all_param

    def print_trainable_parameters(self,return_option=False):# -> None:
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params, all_param = self.get_nb_trainable_parameters()
        if return_option :
            return f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param}"
        else :
            print(
                f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param}"
            )
