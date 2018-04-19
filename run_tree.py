import dynet as dy
import numpy as np
import dataparser
from Nlidataset import Nlidataset
class TreeRNN(object):
	def __init__(self, model):
		sizey=300
		num_classes=3
		self.W=model.add_parameters((sizey,sizey*2))
		self.W2=model.add_parameters((sizey,sizey))
		self.Wf=model.add_parameters((num_classes, sizey*2))

	def __call__(self, tokens1, transitions1, tokens2, transitions2): 
		Wf=dy.parameter(self.Wf)
		#import pdb;pdb.set_trace()
		val=Wf*dy.concatenate([self.expr_for_tree(tokens1, transitions1),self.expr_for_tree(tokens2, transitions2)])
		return val

	def expr_for_tree(self, tokens, transitions,):
		#import pdb;pdb.set_trace()
		tokens=tokens.reshape(-1, 300)
		h=[]
		token_c=0
		for i in range(len(transitions)):
			if transitions[i]==0:
				#shift
				#W=dy.parameter(self.W)
				#import pdb;pdb.set_trace()
				h.append(dy.inputTensor(tokens[token_c]))
				token_c+=1
			elif transitions[i]==1:
				#reduce
				h1=h.pop()
				h2=h.pop()
				W=dy.parameter(self.W)
				W2=dy.parameter(self.W2)
				newh=dy.tanh(W2*W*dy.concatenate([h1,h2]))
				h.append(newh)
			else:
				break
		final=h.pop()
		return dy.tanh(final)

import sys
sys.stdout.write("Staarting off")
model = dy.Model()
batch_size=64
trainer = dy.AdamTrainer(model)
treernn=TreeRNN(model)
import time
file=open("out.x.2", "w")
#data=Nlidataset("/Users/anhadmohananey/Downloads/snli_1.0/snli_1.0_dev.jsonl","/Users/anhadmohananey/Downloads/glove/glove.6B.300d.txt")
data=Nlidataset("/scratch/am8676/snli_1.0/snli_1.0_train.jsonl","/scratch/am8676/glove.840B.300d.txt")
now=time.time()
for i in range(0, len(data), batch_size):	
	losses=[]
	dy.renew_cg()
	file.write(str(i))
	file.write("-")
	print("Batch Training"+str(i))
	for v in data.ra(i, batch_size):		
		(s1, s2, t1, t2), label=v
		preds=treernn(s1, t1, s2, t2)
		#import pdb;pdb.set_trace()
		loss=dy.pickneglogsoftmax(preds, label)
		losses.append(loss)
		#loss.backward()
		#trainer.update()
	later=time.time()
	difference=int(later-now)
	print(difference)
        file.write(str(difference)+"....")
	file.flush()
	#import pdb;pdb.set_trace()
	batch_loss=dy.esum(losses)/batch_size
	batch_loss.backward()
	trainer.update()
#just the traing for now

