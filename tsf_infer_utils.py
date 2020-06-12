import torch
import numpy as np

from tsf_train_utils import prep_transf_inputs, prep_transf_inputs2
from tsf_utils import get_tk_from_proba

from transformer_translation.translate_sentence import gen_nopeek_mask
from transformer_translation.dataset import IDX_EOS, IDX_SOS

import torch.nn as nn
from einops import rearrange

def get_EOS_indices_old(pred):
    # get indices of instances of IDX_EOS in each row
    idxr, idxc = torch.where(pred == IDX_EOS)
    
    if not any(idxr):
        return [-2 for i in range(pred.shape[0])]

    # greedy search of first appearance
    row_many_idx, row_unique_idx = 0, 0
    idx_eos = []
    while row_unique_idx < pred.shape[0] and row_many_idx < idxr.shape[0]:
        if idxr[row_many_idx].item() > row_unique_idx:
            idx_eos.append(-2)
            row_unique_idx += 1
        if idxr[row_many_idx].item() == row_unique_idx:
            idx_eos.append(idxc[row_many_idx].item())
            row_unique_idx += 1
        else:
            row_many_idx += 1
        if row_many_idx == idxr.shape[0]:
            idx_eos.append(-2)
            break
        
    return idx_eos

def get_EOS_indices(pred):
    idx_eos = []
    for i in range(pred.shape[0]):
        idx = torch.where(pred[i,:] == IDX_EOS)[0]
        if idx.shape[0] == 0:
            idx = -2
        else:
            idx = idx.min().item()
        idx_eos.append(idx)
    return idx_eos

def oos_infer_batched2(model, loader, max_seq_length):
    device = model.embed_tgt.weight.device
    pred_list_lg, tgt_list_lg, tag_list_lg = [], [], []

    for (src, src_key_padding_mask, tgt, tgt_key_padding_mask) in iter(loader):

        # format tags
        tag_list = [to_list_npint64(pop_padding_ts(src[:,[i],:]).flatten().tolist()) for i in range(src.shape[1])]

        src = src[0].to(device)

        pred = IDX_SOS * torch.ones((tgt.shape[1], 1), dtype=torch.long, device=device)
        pred_mask = gen_nopeek_mask(pred.shape[1]).to(device)

        while  not (pred == IDX_EOS).any(1).all() and pred.shape[1] < max_seq_length + 1:
            output = model(src, pred, src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None, tgt_mask=pred_mask)#[[-1],:])
            pred = torch.cat([pred, get_tk_from_proba(output)[:,[-1]]], dim=1)
            pred_mask = gen_nopeek_mask(pred.shape[1]).to(device)

        # format pred sentence output
        idx_eos = get_EOS_indices(pred)
        pred_list = [to_list_npint64(pred[i,:idx_eos[i]].tolist()+[IDX_EOS]) for i in range(pred.shape[0])]

        # format tgt sentence output
        tgt_list = [to_list_npint64(pop_padding_ts(tgt[:,[i],:]).flatten().tolist()) for i in range(tgt.shape[1])]

        # aggregate results
        pred_list_lg += pred_list
        tgt_list_lg += tgt_list
        tag_list_lg += tag_list
    
    return pred_list_lg, tgt_list_lg, tag_list_lg

def oos_infer_batched(model, loader, max_seq_length):
    device = model.embed_tgt.weight.device
    pred_list_lg, tgt_list_lg, tag_list_lg = [], [], []

    for (src, src_key_padding_mask, tgt, tgt_key_padding_mask) in iter(loader):

        src = src.to(device)

        pred = IDX_SOS * torch.ones((tgt.shape[1], 1), dtype=torch.long, device=device)
        pred_mask = gen_nopeek_mask(pred.shape[1]).to(device)

        while  not (pred == IDX_EOS).any(1).all() and pred.shape[1] < max_seq_length + 1:
            output = model(src, pred, src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None, tgt_mask=pred_mask)#[[-1],:])
            pred = torch.cat([pred, get_tk_from_proba(output)[:,[-1]]], dim=1)
            pred_mask = gen_nopeek_mask(pred.shape[1]).to(device)

        # format pred sentence output
        idx_eos = get_EOS_indices(pred)
        pred_list = [to_list_npint64(pred[i,:idx_eos[i]].tolist()+[IDX_EOS]) for i in range(pred.shape[0])]

        # format tgt sentence output
        tgt_list = [to_list_npint64(pop_padding_ts(tgt[:,[i],:]).flatten().tolist()) for i in range(tgt.shape[1])]

        # format tags
        tag_idx = src.nonzero()
        tag_list = [tag_idx[tag_idx[:,1] == i, 2] for i in range(src.shape[1])]

        # aggregate results
        pred_list_lg += pred_list
        tgt_list_lg += tgt_list
        tag_list_lg += tag_list
    
    return pred_list_lg, tgt_list_lg, tag_list_lg

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