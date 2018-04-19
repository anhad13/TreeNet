import dataparser
import numpy as np
from ReadEmbeddings import *
class Nlidataset(object):
    def __init__(self, nliPath, glovePath, transform = None):  
        self.data = dataparser.load_data(nliPath)
        self.paddingElement = ['<s>']
        self.maxSentenceLength = self.maxlength(self.data)
        self.maxTransitionLength= self.maxtransitions(self.data)
        self.vocab = glove2dict(glovePath)

    def __getitem__(self, index):
        s1 = self.pad(self.data[index]['premise_tokens'])
        s2 = self.pad(self.data[index]['hypothesis_tokens'])
        t1 = self.pad_transitions(self.data[index]['premise_transitions'])
        t2 = self.pad_transitions(self.data[index]['hypothesis_transitions'])
        s1 = self.embed(s1)
        s2 = self.embed(s2)
        LABEL_MAP = {
            "entailment": 0,
            "neutral": 1,
            "contradiction": 2,
            # Used in the unlabeled test set---needs to map to some arbitrary label.
            "hidden": 0
        }
        label = LABEL_MAP[self.data[index]['label']]
        return (s1, s2, t1, t2), label
    def ra(self, start, number):
        res=[]
        for i in range(number):
            res.append(self[start+i])
        return res
    def __len__(self):
        return len(self.data)
    def maxtransitions(self, data):
        maxTransitionLength = max([max(len(d['premise_transitions']),len(d['hypothesis_transitions'])) for d in data])
        return maxTransitionLength
    def maxlength(self, data):
        maxSentenceLength = max([max(len(d['premise_tokens']),len(d['hypothesis_tokens'])) for d in data])
        return maxSentenceLength
    def pad_transitions(self, transitions):
        return transitions+ (self.maxTransitionLength-len(transitions))*[2]
    def pad(self, sentence):
        return sentence + (self.maxSentenceLength-len(sentence))*self.paddingElement

    def embed(self, sentence):
        vector = []
        for word in sentence:
            if str(word) in self.vocab:
                vector = np.concatenate((vector, self.vocab[str(word)]), axis=0)
            else:
                vector = np.concatenate((vector, [0]*len(self.vocab['a'])), axis=0)
        return vector