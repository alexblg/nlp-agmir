def get_sent_from_tk(tensor_tk, index2word):
    return [index2word[idx.item()] for idx in tensor_tk]
    
def get_tk_from_proba(model_output):
    return model_output.max(dim=2)[1]
    
def print_nl_pred_vs_tgt(pred, tgt, index2word):
    for row in range(pred.shape[0]):
                
        # print target
        nl_tgt = get_sent_from_tk(
            tgt[row, :]
            ,index2word)
        print('TARGET: ',' '.join(nl_tgt))
                
        # print predictions
        try:
            nl_pred = get_sent_from_tk(
                pred[row, :]
                ,index2word)
            print('PREDICTION: ',' '.join(nl_pred))
        except:
            print('out of vocabulary')
            pass
        print('\n')