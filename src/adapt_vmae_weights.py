# -*- coding: utf-8 -*- # @Author: hao cheng  # @Date: 2024-08-21 05:21:57  # @Last Modified by:   hao cheng  # @Last Modified time: 2024-08-21 05:21:57 # -*- coding: utf-8 -*-
# @Time    : 3/13/23 11:17 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : adapt_vmae_weights.py

# adapt original single-modality vision mae weights for multi-modality cav-mae pretraining initialization, decoder is also included.

import torch
import models
from collections import OrderedDict

encoder_depth = 12 # total of 12 layers, 11 layers

# weights from https://github.com/facebookresearch/mae
mae_full_weigth = torch.load('./pretrained_model/mae_pretrain_vit_base_full.pth')['model']
additional_weight = OrderedDict()
# for key in mae_full_weigth.keys():
#     if 'blocks' in key and 'decoder' not in key:
#         block_id = int(key.split('.')[1])
#         key_var_name = '.'.join(key.split('.')[2:])
#         if block_id <= encoder_depth-1:
#             additional_weight['blocks.' + key[7:]] = mae_full_weigth[key].detach().clone()
#         else:
#             additional_weight['blocks.' + str(block_id-encoder_depth) + '.' + key_var_name] = mae_full_weigth[key].detach().clone()

for block_id in range(12):
    additional_weight['blocks.' + str(block_id) + '.norm1_a.weight'] = mae_full_weigth['blocks.' + str(block_id) + '.norm1.weight'].detach().clone()
    additional_weight['blocks.' + str(block_id) + '.norm1_v.weight'] = mae_full_weigth['blocks.' + str(block_id) + '.norm1.weight'].detach().clone()
    additional_weight['blocks.' + str(block_id) + '.norm1_a.bias'] = mae_full_weigth['blocks.' + str(block_id) + '.norm1.bias'].detach().clone()
    additional_weight['blocks.' + str(block_id) + '.norm1_v.bias'] = mae_full_weigth['blocks.' + str(block_id) + '.norm1.bias'].detach().clone()

    additional_weight['blocks.' + str(block_id) + '.norm2_a.weight'] = mae_full_weigth['blocks.' + str(block_id) + '.norm2.weight'].detach().clone()
    additional_weight['blocks.' + str(block_id) + '.norm2_v.weight'] = mae_full_weigth['blocks.' + str(block_id) + '.norm2.weight'].detach().clone()
    additional_weight['blocks.' + str(block_id) + '.norm2_a.bias'] = mae_full_weigth['blocks.' + str(block_id) + '.norm2.bias'].detach().clone()
    additional_weight['blocks.' + str(block_id) + '.norm2_v.bias'] = mae_full_weigth['blocks.' + str(block_id) + '.norm2.bias'].detach().clone()

# NOTE:这里的两个norm在模型中并不会被更新，所以是否需要导入权重？位置是对比学习之前那个地方
additional_weight['norm_a.weight'] = mae_full_weigth['norm.weight'].detach().clone()
additional_weight['norm_v.weight'] = mae_full_weigth['norm.weight'].detach().clone()
additional_weight['norm_a.bias'] = mae_full_weigth['norm.bias'].detach().clone()
additional_weight['norm_v.bias'] = mae_full_weigth['norm.bias'].detach().clone()

mae_mdl = models.Uni_CMAE(encoder_depth=encoder_depth)

miss, unexpect = mae_mdl.load_state_dict(mae_full_weigth, strict=False)
miss_a, unexpect_a = mae_mdl.load_state_dict(additional_weight, strict=False)

# miss 00-02
mae_mdl.cls_token_a = torch.nn.Parameter(mae_full_weigth['cls_token'].detach().clone())
mae_mdl.cls_token_v = torch.nn.Parameter(mae_full_weigth['cls_token'].detach().clone())
mae_mdl.cls_token_av = torch.nn.Parameter(mae_full_weigth['cls_token'].detach().clone())

# # miss 06 #TODO： 这个也要改，导入mae_st的权重
# mae_mdl.pos_embed_v = torch.nn.Parameter(mae_full_weigth['pos_embed'][:,1:,:].detach().clone())

# # miss 08
# mae_mdl.decoder_pos_embed_v = torch.nn.Parameter(mae_full_weigth['decoder_pos_embed'][:,1:,:].detach().clone())

# miss 09-10
mae_mdl.patch_embed_a.proj.weight = torch.nn.Parameter(torch.sum(mae_full_weigth['patch_embed.proj.weight'], dim=1).unsqueeze(1).detach().clone())
mae_mdl.patch_embed_a.proj.bias = torch.nn.Parameter(mae_full_weigth['patch_embed.proj.bias'].detach().clone())

# # miss 11-12 # TODO：这个需要改，导入mae_st的权重应该ok
# mae_mdl.patch_embed_v.proj.weight = torch.nn.Parameter(mae_full_weigth['patch_embed.proj.weight'].detach().clone())
# mae_mdl.patch_embed_v.proj.bias = torch.nn.Parameter(mae_full_weigth['patch_embed.proj.bias'].detach().clone())

# miss 13-14
mae_mdl.decoder_pred_a.weight = torch.nn.Parameter(mae_full_weigth['decoder_pred.weight'][:256].detach().clone())
mae_mdl.decoder_pred_a.bias = torch.nn.Parameter(mae_full_weigth['decoder_pred.bias'][:256].detach().clone())

# miss 15-16
mae_mdl.decoder_pred_v.weight = torch.nn.Parameter(mae_full_weigth['decoder_pred.weight'].detach().clone())
mae_mdl.decoder_pred_v.bias = torch.nn.Parameter(mae_full_weigth['decoder_pred.bias'].detach().clone())

mae_mdl

torch.save(mae_mdl.state_dict(), 'ori_mae_{:d}_for_pretrain.pth'.format(encoder_depth))

new_weigth = 'ori_mae_12_for_pretrain.pth'
new = torch.load(new_weigth)
mae_mdl = models.Uni_CMAE(encoder_depth=encoder_depth)
miss, unexpect = mae_mdl.load_state_dict(new, strict=False)
print(miss, unexpect)