def get_sent_from_tk(tensor_tk, index2word):
    return [index2word[idx.item()] for idx in tensor_tk]

def get_sent_from_tk2(tensor_tk, index2word):
    return [index2word[idx] for idx in tensor_tk]
    
def get_tk_from_proba(model_output):
    return model_output.max(dim=2)[1]
    
def print_nl_pred_vs_tgt(pred, tgt, index2word, tags=None, tags_index2word=None):
    for row in range(len(pred)):
        
        # print target
        if tags:
            nl_tags = get_sent_from_tk2(
                tags[row]
                ,tags_index2word)
            print('TAGS: ',' '.join(nl_tags))
        
        # print target
        nl_tgt = get_sent_from_tk2(
            tgt[row]
            ,index2word)
        print('TARGET: ',' '.join(nl_tgt))
                
        # print predictions
        try:
            nl_pred = get_sent_from_tk2(
                pred[row]
                ,index2word)
            print('PREDICTION: ',' '.join(nl_pred))
        except:
            print('out of vocabulary')
            pass
        print('\n')
        
def format_list_for_bleu(candidate_corpus, references_corpus):
    return [[str(i) for i in sent] for sent in candidate_corpus], [[sent] for sent in [[str(i) for i in sent] for sent in references_corpus]]

import tsf_infer_utils
from torchtext.data.metrics import bleu_score
from tsf_utils import format_list_for_bleu

def get_bleu_from_loader2(model, loader):
    pred_list, tgt_list, _ = tsf_infer_utils.infer2(model, loader)
    pred_list_bleu, tgt_list_bleu = format_list_for_bleu(pred_list, tgt_list)
    return bleu_score(pred_list_bleu, tgt_list_bleu)

def get_bleu_from_loader(model, loader):
    pred_list, tgt_list, _ = tsf_infer_utils.infer(model, loader)
    pred_list_bleu, tgt_list_bleu = format_list_for_bleu(pred_list, tgt_list)
    return bleu_score(pred_list_bleu, tgt_list_bleu)

import datetime
import torch

def save_model(model, model_name, date_format='%Y%m%d_%H%M'):
    filename = 'models/{}_model_{}.pth'.format(
        datetime.datetime.now().strftime(date_format)
        ,model_name
    )
    torch.save(model.state_dict(), filename)
    print('model saved at '+filename)