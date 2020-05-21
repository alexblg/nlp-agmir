import torch
import random

from loaders import normalizeString, tensorFromSentence, EOS_token, SOS_token

def evaluate(encoder, decoder, sentence, max_length, input_lang, output_lang):
    device = encoder.embedding.weight.device
    
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence, device, max_length)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(
                input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]

def evaluateRandomly(ds, encoder, decoder, max_length, input_lang, output_lang, n=10):
    tuples = []
    
    for i in range(n):
        report_id = random.choice(range(len(ds)))
        report_pair = ds.__getitem__(report_id)[:2]
        pair = (report_pair[0], normalizeString(report_pair[1]))
        
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0], max_length, input_lang, output_lang)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')
        tuples.append((pair[0], pair[1], output_sentence))
        
    return tuples

def evaluateAll(ds, encoder, decoder, max_length, input_lang, output_lang):
    tuples = [] 
    
    for report_id in range(len(ds)):
        # get+process pair
        report_pair = ds.__getitem__(report_id)[:2]
        pair = (report_pair[0], normalizeString(report_pair[1]))
        
        # perform inference
        output_words, attentions = evaluate(encoder, decoder, pair[0], max_length, input_lang, output_lang)
        output_sentence = ' '.join(output_words)
        tuples.append((pair[0], pair[1], output_sentence))
        
    return tuples