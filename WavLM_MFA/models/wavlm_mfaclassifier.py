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
from transformers import WavLMModel
from speechbrain.lobes.models.ECAPA_TDNN import AttentiveStatisticsPooling

'''    
def layer_freeze(wavlm, total_layer,frz_layer) :
    
    for para in wavlm.parameters():
        para.requires_grad = False

    for name ,child in wavlm.named_children():
        if name == 'encoder':
            for nam,chil in child.named_children():
                if nam == 'layers':
                    for na,chi in chil.named_children():
                        if frz_layer == -1 :
                            for para in chi.parameters():
                                para.requires_grad = True
                        elif frz_layer <= int(na) :
                            for para in chi.parameters():
                                para.requires_grad = True
    return wavlm                                                
'''

class Model(nn.Module):

    def __init__(self,d_args):

        super(Model, self).__init__()
        '''input : (B,1,F,T)'''

        version = d_args["version"]
        self.wavlm = WavLMModel.from_pretrained(version)
        
        self.asp1 = AttentiveStatisticsPooling(channels=768, attention_channels=128, global_context=True)
        self.bn = nn.BatchNorm1d(18432)
        self.fc1 = nn.Linear(18432, 768)
        self.asp2 = AttentiveStatisticsPooling(channels=768, attention_channels=128, global_context=True)
        self.fc2 = nn.Linear(1536, 2)
        self.relu = nn.ReLU()
        
    def forward(self, x, Freq_aug=False):

        wavlm_out = self.wavlm(x, output_attentions=True, output_hidden_states=True)
        wavlm_encoder_outputs = wavlm_out.hidden_states  
        saved_hidden_states = []
        for i, hidden_state in enumerate(wavlm_encoder_outputs):
            if i > 0:  # 입력 임베딩을 제외한 레이어의 출력만 저장
                saved_hidden_states.append(hidden_state)
                
        # wavlm_out = torch.transpose(wavlm_out,1,2)    # output : (1280,time)
#        print('w2v2 : ', w2v2_out.shape)
        
        pooled_outputs = []
        for hidden_state in saved_hidden_states:
            pooled_output = self.asp1(hidden_state.transpose(1, 2))  # (B, C, L)
            pooled_outputs.append(pooled_output)
        
        concatenated_output = torch.cat(pooled_outputs, dim=2) # 어떤 차원으로 결합?
        concatenated_output = concatenated_output.view(concatenated_output.size(0), -1)
        concatenated_output = self.bn(concatenated_output)
        
        x1 = self.fc1(concatenated_output)
        x1 = self.relu(x1)
        
        x1 = x1.unsqueeze(-1)  # (B, 768) -> (B, 768, 1)
        
        x2 = self.asp2(x1)
        last_hidden = x2.view(x2.size(0), -1)  # Flatten to (B, 768*2)
        
        output = self.fc2(last_hidden)        


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