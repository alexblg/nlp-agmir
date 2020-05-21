import json
import os
from torch.utils.data import Dataset
import unicodedata
import re
import torch

def clean_merge_tags(tags):
    return ' '.join(
        [t.lower().replace(' ','_').replace(',','_').replace('-','_') for t in tags])

class IUXRay(Dataset):
    def __init__(self, caption_json, tags_vocab=None):
        self.cases = self.load_json(caption_json)
        self.case_list = [case['id'] for case in self.cases]
        if tags_vocab:
            self.tags_vocab = tags_vocab
        else:
            self.tags_vocab = self.create_tags_voc()
        
    def __len__(self):
        return len(self.case_list)
        
    def __getitem__(self, index):
        case = self.cases[index]
        if not case['impression']:
            descr = case['findings']
        elif not case['findings']:
            descr = case['impression']
        else:
            descr = case['impression']+' '+case['findings']
        return clean_merge_tags(case['tags_mti']), descr, case['id'], index
        
    def create_tags_voc(self):
        tags = []
        for case in self.cases:
            tags += case['tags_mti']
        tags_voc = [t.lower().replace(' ','_').replace(',','_').replace('-','_') for t in tags]
        return list(set(tags_voc))
        
    def load_json(self, caption_json):
        with open(caption_json,'r') as handle:
            cases = json.load(handle)
        return cases
    
# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

SOS_token = 0
EOS_token = 1

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def getReports():
    print("read data...")

    # load dataset
    data_path = r"/home/alex/data/nlp/agmir"
    from loaders import IUXRay
    ds = IUXRay(os.path.join(data_path,'cases_clean2.json'))

    # make Lang instances
    input_lang = Lang('tags')
    output_lang = Lang('report')

    return input_lang, output_lang, ds

def prepareReportData():
    input_lang, output_lang, ds = getReports()
    
    print("read %s reports" % len(ds))
    
    #pairs = filterPairs(pairs)
    print("trimmed to %s sentence pairs" % len(ds))
    
    print("counting words...")
    for i, report in enumerate(ds):
        #print(i, report)
        input_lang.addSentence(report[0])
        output_lang.addSentence(
            normalizeString(report[1])) # FIX LATER: store normalized reports instead
    print("counted words:")
    print('\t',input_lang.name, input_lang.n_words)
    print('\t',output_lang.name, output_lang.n_words)
    return input_lang, output_lang, ds

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def tensorFromSentence(lang, sentence, device, max_length):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)[:max_length]

def tensorsFromPair(pair, input_lang, output_lang, device, max_length):
    input_tensor = tensorFromSentence(input_lang, pair[0], device, max_length)
    target_tensor = tensorFromSentence(output_lang, pair[1], device, max_length)
    return (input_tensor, target_tensor)