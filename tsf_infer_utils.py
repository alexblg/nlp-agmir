import torch
import numpy as np

from tsf_train_utils import prep_transf_inputs, prep_transf_inputs2
from tsf_utils import get_tk_from_proba

from transformer_translation.translate_sentence import gen_nopeek_mask
from transformer_translation.dataset import IDX_EOS, IDX_SOS

import torch.nn as nn
from einops import rearrange

def forward_model(model, src, pred_sentence, device):
    tgt = torch.tensor(pred_sentence).unsqueeze(0).unsqueeze(0).to(device)

    # prepare inputs
    src = src.to(device)
    tgt = tgt[0].to(device)
    tgt_mask = gen_nopeek_mask(tgt.shape[1]).to(device)

    # run inference
    return model(src, tgt, src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None, tgt_mask=tgt_mask)

def oos_infer_sent(model, src, tgt, max_seq_length, device): 
    # replace by initial tgt for out-of-sample decoding
    pred = [IDX_SOS]

    # run out-of-sample decoding
    i = 0
    while int(pred[-1]) != IDX_EOS and i < max_seq_length:
        output = get_tk_from_proba(
            forward_model(model, src, pred, device))
        pred.append(output[0][-1].item())
        i += 1
        
    # format outputs
    pred_list, tgt_list = [to_list_npint64(pred)], [to_list_npint64(pop_padding_ts(tgt).flatten().tolist())]
    tag_list = [src.nonzero()[:,2]]
    
    return pred_list, tgt_list, tag_list

def oos_infer(model, loader, max_seq_length):
    pred_list, tgt_list, tag_list = [], [], []
    device = model.embed_tgt.weight.device

    # run through batches
    for (src, src_key_padding_mask, tgt, tgt_key_padding_mask) in iter(loader):

        # run through sentences in batch
        for i in range(src.shape[1]):

            # select single sentence
            src_i = src[:,[i],:]
            tgt_i = tgt[:,[i],:]

            pred_out, tgt_out, tag_out = oos_infer_sent(model, src_i, tgt_i, max_seq_length, device)
            pred_list += pred_out
            tgt_list += tgt_out
            tag_list += tag_out
    return pred_list, tgt_list, tag_list

def pop_padding(tk_list):
    while tk_list[-2:] == [0, 0]:
        tk_list.pop(-1)
    tk_list.pop(-1)
    return tk_list

def pop_padding_ts(tk_tensor):
    return tk_tensor[:,:,tk_tensor.nonzero()[:,2]]

def to_list_npint64(list_int):
    return [np.int64(tk) for tk in list_int]

criterion = nn.CrossEntropyLoss(ignore_index=0)

def infer2(model, loader):
    pred_list, tgt_list = [], []
    loss_per_batch = []
    device = model.embed_tgt.weight.device
    
    for (src, src_key_padding_mask, tgt, tgt_key_padding_mask) in iter(loader):
        
        # prepare inputs
        src, src_key_padding_mask, tgt, tgt_key_padding_mask, memory_key_padding_mask, tgt_inp, tgt_out, tgt_mask = prep_transf_inputs2(
            src, src_key_padding_mask, tgt, tgt_key_padding_mask, device)

        # run inference
        outputs = model(src, tgt_inp, src_key_padding_mask, tgt_key_padding_mask[:, :-1], memory_key_padding_mask, tgt_mask)
        loss = criterion(rearrange(outputs, 'b t v -> (b t) v'), rearrange(tgt_out, 'b o -> (b o)'))

        # get predictions from proba
        pred = get_tk_from_proba(outputs)
        
        # get pred and ground truth ready for metric eval
        pred_list += [list(pred[row, :].cpu().numpy()) for row in range(pred.shape[0])]
        tgt_list += [list(tgt[row, :].cpu().numpy()) for row in range(pred.shape[0])]
        
        # record loss
        loss_per_batch += [loss.item()]
    return pred_list, tgt_list, loss_per_batch

def infer(model, loader):
    pred_list, tgt_list = [], []
    loss_per_batch = []
    device = model.embed_tgt.weight.device
    
    for (src, src_key_padding_mask, tgt, tgt_key_padding_mask) in iter(loader):
        
        # prepare inputs
        src, src_key_padding_mask, tgt, tgt_key_padding_mask, memory_key_padding_mask, tgt_inp, tgt_out, tgt_mask = prep_transf_inputs(
            src, src_key_padding_mask, tgt, tgt_key_padding_mask, device)

        # run inference
        outputs = model(src, tgt_inp, src_key_padding_mask, tgt_key_padding_mask[:, :-1], memory_key_padding_mask, tgt_mask)
        loss = criterion(rearrange(outputs, 'b t v -> (b t) v'), rearrange(tgt_out, 'b o -> (b o)'))

        # get predictions from proba
        pred = get_tk_from_proba(outputs)
        
        # get pred and ground truth ready for metric eval
        pred_list += [list(pred[row, :].cpu().numpy()) for row in range(pred.shape[0])]
        tgt_list += [list(tgt[row, :].cpu().numpy()) for row in range(pred.shape[0])]
        
        # record loss
        loss_per_batch += [loss.item()]
    return pred_list, tgt_list, loss_per_batch