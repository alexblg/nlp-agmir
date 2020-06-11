import pickle
import random
import numpy as np

from torch.utils.data import Dataset
from sklearn.feature_extraction.text import CountVectorizer

IDX_SOS, IDX_EOS, IDX_PAD = 1, 2, 0

class TagReportDataset(Dataset):
    def __init__(self, tags_path, reports_path, num_tokens, max_seq_length, idxs=None, countvec=None):
        self.num_tokens = num_tokens
        
        # load data
        self.tags, self.reports, self.data_lengths = load_tags_reports(tags_path, reports_path, max_seq_length, idxs=idxs)
        
        # create count vectorizer
        if countvec:
            self.countvec = countvec
        else:
            self.countvec = create_countvec(self.tags)
        
        # generate batches
        self.batches = gen_batches_tags_reports(num_tokens, self.data_lengths)

    def __getitem__(self, idx):
        tgt, tgt_mask = getitem_report(idx, self.reports, self.batches)
        src_bow, src_bow_masks = getitem_tags(idx, self.tags, self.batches, self.countvec)

        return src_bow.toarray(), src_bow_masks, tgt, tgt_mask

    def __len__(self):
        return len(self.batches)

    def shuffle_batches(self):
        self.batches = gen_batches_tags_reports(self.num_tokens, self.data_lengths)
        
def create_countvec(tags):
        tag_vocab = list(set([j for i in tags for j in i.split()]))
        return CountVectorizer(vocabulary=tag_vocab)

def getitem_tags(idx, data, batches, countvec):
    sentence_indices = batches[idx]
    batch = [data[i] for i in sentence_indices]
    
    return countvec.transform(batch), np.array([[False] for i in range(len(batch))])
    
def getitem_report(idx, data, batches):
    sentence_indices = batches[idx]
    batch = [[IDX_SOS] + data[i] + [IDX_EOS] for i in sentence_indices]

    seq_length = 0
    for sentence in batch:
        if len(sentence) > seq_length:
            seq_length = len(sentence)

    masks = []
    for i, sentence in enumerate(batch):
        masks.append([False for _ in range(len(sentence))] + [True for _ in range(seq_length - len(sentence))])
        batch[i] = sentence + [IDX_PAD for _ in range(seq_length - len(sentence))]

    return np.array(batch), np.array(masks)

def gen_batches_tags_reports(num_tokens, data_lengths):
    # Shuffle all the indices
    for k, v in data_lengths.items():
        random.shuffle(v)

    batches = []
    prev_tokens_in_batch = 1e10
    for k in sorted(data_lengths):
        v = data_lengths[k]
        total_tokens = k * len(v)

        while total_tokens > 0:
            tokens_in_batch = min(total_tokens, num_tokens) - min(total_tokens, num_tokens) % (k)
            sentences_in_batch = tokens_in_batch // (k)

            # Combine with previous batch?
            if tokens_in_batch + prev_tokens_in_batch <= num_tokens:
                batches[-1].extend(v[:sentences_in_batch])
                prev_tokens_in_batch += tokens_in_batch
            else:
                batches.append(v[:sentences_in_batch])
                prev_tokens_in_batch = tokens_in_batch
            v = v[sentences_in_batch:]

            total_tokens = k * len(v)
    return batches

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
    
def load_report_voc(report_voc_path):
    reports_word2index = load_pickle(report_voc_path)
    reports_index2word = dict(zip(reports_word2index.values(), reports_word2index.keys()))
    reports_index2word[IDX_SOS], reports_index2word[IDX_EOS] = "SOS", "EOS"
    return reports_index2word

def load_tags_reports(tags_path, reports_path, max_seq_length, idxs=None):
    # load tags and reports
    tags = load_pickle(tags_path)
    reports = load_pickle(reports_path)
    
    # only select examples in the index
    if idxs:
        tags = [tags[i] for i in idxs]
        reports = [reports[i] for i in idxs]

    # group reports according to their exact length, and discard some reports
    data_lengths = {}
    for i, str_ in enumerate(reports):
        # discard reports with no text or text of length > max_seq_length
        if 0 < len(str_) <= max_seq_length - 2: # -2 since we add 1 token at beg of sequence, 1 at the end
            if len(str_) in data_lengths:
                data_lengths[len(str_)].append(i)
            else:
                data_lengths[len(str_)] = [i]
    return tags, reports, data_lengths

class ParallelLanguageDataset(Dataset):
    def __init__(self, data_path_1, data_path_2, num_tokens, max_seq_length, idxs=None):
        self.num_tokens = num_tokens
        self.data_1, self.data_2, self.data_lengths = load_data(data_path_1, data_path_2, max_seq_length, idxs)

        self.batches = gen_batches(num_tokens, self.data_lengths)

    def __getitem__(self, idx):
        src, src_mask = getitem(idx, self.data_1, self.batches, True)
        tgt, tgt_mask = getitem(idx, self.data_2, self.batches, False)

        return src, src_mask, tgt, tgt_mask

    def __len__(self):
        return len(self.batches)

    def shuffle_batches(self):
        self.batches = gen_batches(self.num_tokens, self.data_lengths)


def gen_batches(num_tokens, data_lengths):
    # Shuffle all the indices
    for k, v in data_lengths.items():
        random.shuffle(v)

    batches = []
    prev_tokens_in_batch = 1e10
    for k in sorted(data_lengths):
        v = data_lengths[k]
        total_tokens = (k[0] + k[1]) * len(v)

        while total_tokens > 0:
            tokens_in_batch = min(total_tokens, num_tokens) - min(total_tokens, num_tokens) % (k[0] + k[1])
            sentences_in_batch = tokens_in_batch // (k[0] + k[1])

            # Combine with previous batch?
            if tokens_in_batch + prev_tokens_in_batch <= num_tokens:
                batches[-1].extend(v[:sentences_in_batch])
                prev_tokens_in_batch += tokens_in_batch
            else:
                batches.append(v[:sentences_in_batch])
                prev_tokens_in_batch = tokens_in_batch
            v = v[sentences_in_batch:]

            total_tokens = (k[0] + k[1]) * len(v)
    return batches


def load_data(data_path_1, data_path_2, max_seq_length, idxs=None):
    # load sequences
    data_1 = load_pickle(data_path_1)
    data_2 = load_pickle(data_path_2)
    
    # only select examples in the index
    if idxs:
        data_1 = [data_1[i] for i in idxs]
        data_2 = [data_2[i] for i in idxs]
    
    data_lengths = {}
    for i, (str_1, str_2) in enumerate(zip(data_1, data_2)):
        if 0 < len(str_1) <= max_seq_length and 0 < len(str_2) <= max_seq_length - 2: # -2 since we add 1 token at beg of sequence, 1 at the end
            if (len(str_1), len(str_2)) in data_lengths:
                data_lengths[(len(str_1), len(str_2))].append(i)
            else:
                data_lengths[(len(str_1), len(str_2))] = [i]
    return data_1, data_2, data_lengths


def getitem(idx, data, batches, src):
    sentence_indices = batches[idx]
    if src:
        batch = [data[i] for i in sentence_indices]
    else:
        batch = [[IDX_SOS] + data[i] + [IDX_EOS] for i in sentence_indices]

    seq_length = 0
    for sentence in batch:
        if len(sentence) > seq_length:
            seq_length = len(sentence)

    masks = []
    for i, sentence in enumerate(batch):
        masks.append([False for _ in range(len(sentence))] + [True for _ in range(seq_length - len(sentence))])
        batch[i] = sentence + [IDX_PAD for _ in range(seq_length - len(sentence))]

    return np.array(batch), np.array(masks)
