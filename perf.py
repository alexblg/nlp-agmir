from nltk.translate.bleu_score import sentence_bleu
import numpy as np

def get_av_bleu(pred_output):
    return np.mean([
        sentence_bleu([r[1].split(' ')], r[2].split(' ')
                     ) for r in pred_output])