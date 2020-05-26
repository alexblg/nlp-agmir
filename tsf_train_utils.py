from transformer_translation.translate_sentence import gen_nopeek_mask

def prep_transf_inputs2(src, src_key_padding_mask, tgt, tgt_key_padding_mask, device):
    src, src_key_padding_mask = src[0].to(device), src_key_padding_mask[0].to(device)
    tgt, tgt_key_padding_mask = tgt[0].to(device), tgt_key_padding_mask[0].to(device)

    memory_key_padding_mask = src_key_padding_mask.clone()
    tgt_inp, tgt_out = tgt[:, :-1], tgt[:, 1:]
    tgt_mask = gen_nopeek_mask(tgt_inp.shape[1]).to(device)
    
    return src, src_key_padding_mask, tgt, tgt_key_padding_mask, memory_key_padding_mask, tgt_inp, tgt_out, tgt_mask

def prep_transf_inputs(src, src_key_padding_mask, tgt, tgt_key_padding_mask, device):
    src, src_key_padding_mask = src.to(device), src_key_padding_mask[0].to(device)
    tgt, tgt_key_padding_mask = tgt[0].to(device), tgt_key_padding_mask[0].to(device)

    memory_key_padding_mask = src_key_padding_mask.clone()
    tgt_inp, tgt_out = tgt[:, :-1], tgt[:, 1:]
    tgt_mask = gen_nopeek_mask(tgt_inp.shape[1]).to(device)
    
    return src, src_key_padding_mask, tgt, tgt_key_padding_mask, memory_key_padding_mask, tgt_inp, tgt_out, tgt_mask