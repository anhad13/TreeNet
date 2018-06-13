import dynet as dy
import numpy as np
import csv
import json
import argparse
import re
import codecs

def read_dataset(filename, vocab):
    dp=load_data(filename)
    return [read_element(line, vocab) for line in dp]
    #return [line for line in dp]

LABEL_MAP = {
    "entailment": 0,
    "neutral": 1,
    "contradiction": 2,
    # Used in the unlabeled test set---needs to map to some arbitrary label.
    "hidden": 0,
}


def convert_binary_bracketing(parse, lowercase=False):
    transitions = []
    tokens = []

    for word in parse.split(' '):
        if word[0] != "(":
            if word == ")":
                transitions.append(1)
            else:
                # Downcase all words to match GloVe.
                if lowercase:
                    tokens.append(word.lower())
                else:
                    tokens.append(word)
                transitions.append(0)
    return tokens, transitions


def load_data(path, lowercase=False, choose=lambda x: True, eval_mode=False, level="all"):
    print "Loading", path
    examples = []
    failed_parse = 0
    with codecs.open(path, encoding='utf-8') as f:
        for line in f:
            try:
                line = line.encode('UTF-8')
            except UnicodeError as e:
                print "ENCODING ERROR:", line, e
                line = "{}"
            loaded_example = json.loads(line)
            if loaded_example["gold_label"] not in LABEL_MAP:
                continue
            if not choose(loaded_example):
                continue
            example = {}
            example["label"] = loaded_example["gold_label"]
            example["premise"] = loaded_example["sentence1"]
            example["hypothesis"] = loaded_example["sentence2"]
            example["example_id"] = loaded_example.get('pairID', 'NoID')
            if loaded_example["sentence1_binary_parse"] and loaded_example["sentence2_binary_parse"]:
                (example["premise_tokens"], example["premise_transitions"]) = convert_binary_bracketing(
                    loaded_example["sentence1_binary_parse"], lowercase=lowercase)
                (example["hypothesis_tokens"], example["hypothesis_transitions"]) = convert_binary_bracketing(
                    loaded_example["sentence2_binary_parse"], lowercase=lowercase)
                examples.append(example)
            else:
                failed_parse += 1
    if failed_parse > 0:
        print(
            "Warning: Failed to convert binary parse for {} examples.".format(failed_parse))
    return examples


def glove2dict(src_filename):
    reader = csv.reader(open(src_filename), delimiter=' ', quoting=csv.QUOTE_NONE)
    return {line[0]: np.array(list(map(float, line[1: ]))) for line in reader}

def read_element(line, vocab):
    LABEL_MAP = {
    "entailment": 0,
    "neutral": 1,
    "contradiction": 2,
    # Used in the unlabeled test set---needs to map to some arbitrary label.
    "hidden": 0,
    }
    return (converter2(line["premise_tokens"], vocab), line["premise_transitions"], converter2(line["hypothesis_tokens"], vocab), line["hypothesis_transitions"], LABEL_MAP[line['label']])



def converter2(tokens, vocab):
    h=[]
    ftokens=[]
    for i in range(len(tokens)):
        if tokens[i].lower() in vocab:
            val=vocab[tokens[i].lower()]
        else:
            #val=0
            vocab[tokens[i].lower()]=len(vocab)
            val=len(vocab)
        ftokens.append(val)
    return ftokens


class TreeLSTMWO(object):
    def __init__(self, model, vocab_size, wdim, hdim):
        self.WS = [model.add_parameters((hdim, wdim)) for _ in "iou"]
        self.US = [model.add_parameters((hdim, 2*hdim)) for _ in "iou"]
        self.UFS =[model.add_parameters((hdim, hdim)) for _ in "ff"]
        self.BS = [model.add_parameters(hdim) for _ in "iouf"]
        self.E = model.add_lookup_parameters((vocab_size , wdim))
    def expr_for_tree(self, tokens, transitions, decorate=False):
        ha=[]
        ca=[]
        token_n=0
        for i in range(len(transitions)):
            if transitions[i]==0:#shift
                E = self.E#dy.parameter(self.E)
                emb=E[tokens[token_n]]
                token_n+=1
                Wi,Wo,Wu   = [dy.parameter(w) for w in self.WS]
                bi,bo,bu,_ = [dy.parameter(b) for b in self.BS]
                i = dy.logistic(bi+Wi*emb)
                o = dy.logistic(bo+Wo*emb)
                u = dy.tanh(    bu+Wu*emb)
                c = dy.cmult(i,u)
                h = dy.cmult(o,dy.tanh(c))
                ha.append(h)
                ca.append(c)
            elif transitions[i]==1:#reduce
                c1=ca.pop()
                c2=ca.pop()
                e1=ha.pop()
                e2=ha.pop()
                Ui,Uo,Uu = [dy.parameter(u) for u in self.US]
                Uf1,Uf2 = [dy.parameter(u) for u in self.UFS]
                bi,bo,bu,bf = [dy.parameter(b) for b in self.BS]
                e = dy.concatenate([e1,e2])
                i = dy.logistic(bi+Ui*e)
                o = dy.logistic(bi+Uo*e)
                f1 = dy.logistic(bf+Uf1*e1)
                f2 = dy.logistic(bf+Uf2*e2)
                u = dy.tanh(     bu+Uu*e)
                c = dy.cmult(i,u) + dy.cmult(f1,c1) + dy.cmult(f2,c2)
                h = dy.cmult(o,dy.tanh(c))
                ha.append(h)
                ca.append(c)
            else:
                print("Invalid")
        return ha.pop(), ca.pop()


import sys
sys.stdout.write("Staarting off")
#glovePath="/Users/anhadmohananey/Downloads/glove/glove.6B.300d.txt"
glovePath="/scratch/am8676/glove.840B.300d.txt"
vocab = glove2dict(glovePath)
c=1
vocab_embeddings=[]
vocab_embeddings.append(np.zeros(300))
for k in vocab.keys():
    vocab_embeddings.append(vocab[k])
    vocab[k]=c
    c+=1
model = dy.Model()
batch_size=64
eval_every=batch_size*100
#import pdb;pdb.set_trace()
training_data=read_dataset("/scratch/am8676/snli_1.0/snli_1.0_train.jsonl", vocab)
dev_data=read_dataset("/scratch/am8676/snli_1.0/snli_1.0_dev.jsonl", vocab)
#training_data=read_dataset("/Users/anhadmohananey/Downloads/snli_1.0/snli_1.0_train.jsonl", vocab)
#pdb.set_trace()
#dev_data=read_dataset("/Users/anhadmohananey/Downloads/snli_1.0/snli_1.0_dev.jsonl", vocab)
#pdb.set_trace()
number_unk=len(vocab)-len(vocab_embeddings)
for i in range(number_unk):
    vocab_embeddings.append(np.zeros(300))#set all UNK to 0. fine tune *should* take care of the rest.
trainer = dy.AdamTrainer(model, 0.0003)
treernn = TreeLSTMWO(model, len(vocab_embeddings), 300, 300)
#import pdb;pdb.set_trace()
treernn.E.init_from_array(np.array(vocab_embeddings))

WfU=model.add_parameters((300, 300*2))
WfU2=model.add_parameters((3,300))
import time
dy.renew_cg()
start_time=time.time()
filename=open("dynetsnli-1.0"+str(start_time), "w")
losses=[]
no_epochs=5
for epoch_number in range(no_epochs):
    dy.renew_cg()
    losses=[]
    for i in range(len(training_data)):
        Wf=dy.parameter(WfU)
        Wf2=dy.parameter(WfU2)
        d=training_data[i]
        t1, _=treernn.expr_for_tree(d[0], d[1])
        t2, _=treernn.expr_for_tree(d[2], d[3])
        preds_1=Wf*dy.concatenate([t1,t2])
        preds_1=dy.tanh(preds_1)
        preds=Wf2*preds_1
        losses.append(dy.pickneglogsoftmax(preds, d[4]))
        if i>0 and i%batch_size==0:
            batch_loss=dy.esum(losses)/len(losses)
            print("batch done")
            difference=time.time()-start_time
            filename.write(str(i)+":"+str(difference)+"--")
            filename.write("\n")
            filename.flush()
            losses=[]
            batch_loss.backward()
            trainer.update()
            dy.renew_cg()
        if i>0 and i%eval_every==0:
            print("Running eval!")
            results=[]
            actual_results=[]
            correct=0.0
            for j in range(len(dev_data)):
                Wf=dy.parameter(WfU)
                d=dev_data[j]
                t1, _=treernn.expr_for_tree(d[0], d[1])
                t2, _=treernn.expr_for_tree(d[2], d[3])
                preds_1=Wf*dy.concatenate([t1,t2])
                preds_1=dy.tanh(preds_1)
                preds=Wf2*preds_1
                results.append(preds)
                actual_results.append(d[4])
                if j>0 and j%batch_size==0:
                    predictions=np.array([np.argmax(act_value.value()) for act_value in results])
                    actual_results=np.array(actual_results)
                    correct+=np.sum(actual_results==predictions)
                    dy.renew_cg()
                    actual_results=[]
                    results=[]
            print("Dev accuracy:"+ str(float(correct)/len(dev_data)))
            filename.write("Dev accuracy:"+ str(float(correct)/len(dev_data)))
            filename.write("\n")
            filename.flush()