import json
from torch.utils.data import Dataset

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