from tsf_train_utils import prep_transf_inputs
from tsf_utils import get_tk_from_proba

def infer(model, loader):
    pred_list = []
    tgt_list = []
    device = model.embed_tgt.weight.device
    
    for (src, src_key_padding_mask, tgt, tgt_key_padding_mask) in iter(loader):
        
        # prepare inputs
        src, src_key_padding_mask, tgt, tgt_key_padding_mask, memory_key_padding_mask, tgt_inp, tgt_out, tgt_mask = prep_transf_inputs(
            src, src_key_padding_mask, tgt, tgt_key_padding_mask, device)

        # run inference
        outputs = model(src, tgt_inp, src_key_padding_mask, tgt_key_padding_mask[:, :-1], memory_key_padding_mask, tgt_mask)

        # get predictions from proba
        pred = get_tk_from_proba(outputs)
        
        # get pred and ground truth ready for metric eval
        pred_list += [list(pred[row, :].cpu().numpy()) for row in range(pred.shape[0])]
        tgt_list += [list(tgt[row, :].cpu().numpy()) for row in range(pred.shape[0])]
    return pred_list, tgt_list