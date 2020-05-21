import torch

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

# load data
from loaders import prepareReportData

input_lang, output_lang, ds = prepareReportData()

# instantiate models
from models import EncoderRNN, AttnDecoderRNN

hidden_size = 256
max_length = 30
encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, max_length, dropout_p=0.1).to(device)

# train
from train_utils import trainIters

print("\ntraining on {}..".format(device))
n_iter = 500000
print_every = 10000
plot_losses = trainIters(ds, encoder1, attn_decoder1, n_iter, max_length, input_lang, output_lang, print_every=print_every)

# eval
from perf import get_av_bleu
from test_utils import evaluateAll

print("\nevaluating on all images..")
pred_output = evaluateAll(ds, encoder1, attn_decoder1, max_length, input_lang, output_lang)
print('BLEU: {:.2%}'.format(
    get_av_bleu(pred_output)))