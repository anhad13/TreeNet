import dynet as dy
import numpy as np
import dataparser
import json
from Nlidataset import Nlidataset
import argparse
import re
import codecs
from ReadEmbeddings import *

def read_dataset(filename, vocab):
    dp=dataparser.load_data(filename)
    return [read_element(line, vocab) for line in dp]

LABEL_MAP = {
    "entailment": 0,
    "neutral": 1,
    "contradiction": 2,
    # Used in the unlabeled test set---needs to map to some arbitrary label.
    "hidden": 0,
}

def read_element(line, vocab):
    LABEL_MAP = {
    "entailment": 0,
    "neutral": 1,
    "contradiction": 2,
    # Used in the unlabeled test set---needs to map to some arbitrary label.
    "hidden": 0,
    }
    return converter(line["premise_tokens"], line["premise_transitions"],vocab), converter(line["hypothesis_tokens"], line["hypothesis_transitions"], vocab),LABEL_MAP[line["label"]]


def converter(tokens, transitions, vocab):
	h=[]
	tokens_c=0
	for i in range(len(transitions)):
		if transitions[i]==0:
            		if tokens[tokens_c].lower() in vocab:
                		val=vocab[tokens[tokens_c].lower()]
            		else:
                		val=0
			h.append(Tree(val, None))
			tokens_c+=1
		elif transitions[i]==1:
			x1=h.pop()
			x2=h.pop()
			h.append(Tree(None, [x1, x2]))
		else:
			break
	return h.pop()

class Tree(object):
    def __init__(self, label, children=None):
        self.label = label
        self.children = children
    @staticmethod
    def from_sexpr(string):
        toks = iter(_tokenize_sexpr(string))
        assert next(toks) == "("
        return _within_bracket(toks)
    def __str__(self):
        if self.children is None: return self.label
        return "[%s %s]" % (self.label, " ".join([str(c) for c in self.children]))
    def isleaf(self): return self.children==None
    def leaves_iter(self):
        if self.isleaf():
            yield self
        else:
            for c in self.children:
                for l in c.leaves_iter(): yield l
    def leaves(self): return list(self.leaves_iter())
    def nonterms_iter(self):
        if not self.isleaf():
            yield self
            for c in self.children:
                for n in c.nonterms_iter(): yield n
    def nonterms(self): return list(self.nonterms_iter())

class TreeRNN(object):
    def __init__(self, model, vocab_size):
        num_classes=3
        sizey=300
        self.E=model.add_parameters((vocab_size, sizey))
        self.W=model.add_parameters((sizey, sizey*2))
    def expr_for_tree(self, tree):
        if tree.isleaf():
            return self.E[tree.label]
        else:
            e1=self.expr_for_tree(tree.children[0])
            e2=self.expr_for_tree(tree.children[1])
            W=dy.parameter(self.W)
            expr=dy.tanh(W*dy.concatenate([e1,e2]))
            return expr


import sys
sys.stdout.write("Staarting off")
glovePath="/scratch/am8676/glove.840B.300d.txt"

vocab = glove2dict(glovePath)
c=0
for k in vocab.keys():
    vocab[k]=c
    c+=1
model = dy.Model()
batch_size=512
data=read_dataset("/scratch/am8676/snli_1.0/snli_1.0_train.jsonl", vocab)
trainer = dy.AdamTrainer(model)
treernn = TreeRNN(model, len(vocab))
WfU=model.add_parameters((3, 300*2))
import time
dy.renew_cg()
start_time=time.time()
for i in range(0, len(data), batch_size):
    dy.renew_cg()
    dats=data[i:i+batch_size]
    #iterate across batch
    Wf=dy.parameter(WfU)
    losses=[]
    for d in dats:
        t1=treernn.expr_for_tree(d[0])
        t2=treernn.expr_for_tree(d[1])
        preds=Wf*dy.concatenate([t1,t2])
        losses.append(dy.pickneglogsoftmax(preds, d[2]))
    batch_loss=dy.esum(losses)/batch_size
    batch_loss.backward()
    trainer.update()
    difference=time.time()-start_time
    print(str(i)+"---"+str(difference)+":")