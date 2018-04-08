# -*- coding: utf-8 -*-
"""
Neural Network with SGD for handwriting recognition.

Created on Sun Mar 04 17:37:04 2018

@author: Paras Patel
"""

import numpy as np
import sys

data_train=np.matrix(np.genfromtxt(sys.argv[1],dtype=None,delimiter=','))
y=data_train[:,0]
x_1=data_train[:,1:]
x=np.insert(x_1,0,1,axis=1)

data_validate=np.matrix(np.genfromtxt(sys.argv[2],dtype=None,delimiter=','))
y_validate=data_validate[:,0]
x_1_validate=data_validate[:,1:]
x_validate=np.insert(x_1_validate,0,1,axis=1)

train_out=open(sys.argv[3],"w")
validate_out=open(sys.argv[4],"w")
metrics=open(sys.argv[5],"w")

hu=int(sys.argv[7])
epoch=int(sys.argv[6])
epoch_count=0
learningrate=float(sys.argv[9])
flag=int(sys.argv[8])
if(flag==1):
    g_alpha1=np.zeros((hu,x.shape[1]))
    g_beta1=np.zeros((np.size(np.unique(np.array(y))),hu+1))
    alpha=np.random.uniform(-0.1,0.1,size=g_alpha1.shape)
    beta=np.random.uniform(-0.1,0.1,size=g_beta1.shape)
    
if(flag==2):
    g_alpha1=np.zeros((hu,x.shape[1]))
    g_beta1=np.zeros((np.size(np.unique(np.array(y))),hu+1))
    alpha=np.zeros(g_alpha1.shape)
    beta=np.zeros(g_beta1.shape)


def sigmoid(a):
    one=np.ones(a.shape)
    y_s=one/(one+np.exp(-a))
    return y_s

def sigmoidforward(a_f,alpha_f):
    b=sigmoid(a_f)
    return b

def sigmoidbackward(a_b,b_b,g_b):
    e=np.multiply(b_b,1-b_b)
    g_a = np.multiply(g_b,e)
    return g_a

def linearforward(a,alpha_lf):
    b=np.dot(a,np.transpose(alpha_lf))
    return b

def linearbackward(a,omega,g_b):
    g_omega=np.multiply(g_b,np.transpose(a))
    g_a=(g_b*(omega))
    return g_omega,g_a

def softmaxforward(a_sm):
    b_s=np.exp(a_sm)/np.sum(np.exp(a_sm))
    return b_s

def softmaxbackward(a_s,b_s,g_b_s):
    q=np.multiply(b_s,b_s.T)
    w=(np.diag(b_s.A1)-q)
    g_a_s=g_b_s*w
    return g_a_s

def crossentropyforward(a_c,a_c_cap):
    index=np.array(a_c)
    ac=(np.zeros(10))
    ac[index]=1
    b_c=((np.log(a_c_cap)*np.transpose(np.matrix(-ac))))
    return (b_c)

def crossentropybackward(a_c,a_c_cap,b_c,g_b_c):
    index=np.array(a_c)[0][0]
    ac=np.zeros(10)
    ac[index]=1
    g_a_cap=np.multiply(-g_b_c,(np.divide(ac,a_c_cap)))
    return g_a_cap

class forward(object):
    def __init__(self, x,a,z,b,y_cap,j):
        self.a = a
        self.x = x
        self.z = z
        self.b = b
        self.y_cap = y_cap
        self.j = j
        
def NNforward(x_1,y_1,alpha_N,beta_N):
    example_1=x_1
    y_1=y_1
    a=linearforward(example_1,alpha_N)
    z=sigmoidforward(a,alpha_N)
    z=np.insert(z,0,1)
    b=linearforward(z,beta_N)
    y_cap=softmaxforward(b)
    J=crossentropyforward(y_1,y_cap)
    f=forward(example_1,a,z,b,y_cap,J)
    return f

def NNbackward(x,y,alpha_N,beta_N,f):
    y_1 = y
    g_j = np.ones(f.y_cap.shape)
    g_y_cap = crossentropybackward(np.multiply(np.ones(f.y_cap.shape),y_1),f.y_cap,f.j,g_j)
    g_b = softmaxbackward(f.b,f.y_cap,g_y_cap)
    g_beta,g_z = linearbackward(f.z,beta_N,g_b)
    g_a = sigmoidbackward(f.a,np.delete(f.z,0),np.delete(g_z,0))
    g_alpha,g_x = linearbackward(f.x,alpha_N,g_a)
    return g_alpha,g_beta

while(epoch_count<epoch):
    jumbo=0
    jumbo_validate=0
    for i in range(len(data_train)):
        example=x[i]
        label=y[i]
        ob=NNforward(example,label,alpha,beta)
        g_alp,g_bet=NNbackward(example,label,alpha,beta,ob)
        alpha=(alpha)-(learningrate*g_alp.T)
        beta=beta-(learningrate*g_bet.T)
    for i in range(len(data_train)):
        example=x[i]
        label=y[i]
        ok=NNforward(example,label,alpha,beta)
        jumbo=jumbo+np.array(ok.j)[0][0]
    for i in range(len(data_validate)):
        example_validate=x_validate[i]
        label_validate=y_validate[i]
        ok_validate=NNforward(example_validate,label_validate,alpha,beta)
        jumbo_validate=jumbo_validate+np.array(ok_validate.j)[0][0]
    metrics.write("epoch={0} crossentropy(train): {1:.11f}\n".format(epoch_count+1,jumbo/len(data_train)))
    metrics.write("epoch={0} crossentropy(validation): {1:.11f}\n".format(epoch_count+1,jumbo_validate/len(data_validate)))
    epoch_count+=1
    
tr=0
fl=0.0

for i in range(len(data_train)):
    example=x[i]
    label=y[i]
    ok=NNforward(example,label,alpha,beta)
    if(np.argmax(np.array(ok.y_cap)) == np.array(label)[0][0]):
        tr+=1
    else:
        fl+=1
    train_out.write("%d\n" % np.argmax(np.array(ok.y_cap)))
    
tr_val=0
fl_val=0.0

for i in range(len(data_validate)):
    example=x_validate[i]
    label=y_validate[i]
    ok_validate=NNforward(example,label,alpha,beta)
    if(np.argmax(np.array(ok_validate.y_cap)) == np.array(label)[0][0]):
        tr_val+=1
    else:
        fl_val+=1
    validate_out.write("%d\n" % np.argmax(np.array(ok_validate.y_cap)))

metrics.write("error(train): %f\n" % (fl/(tr+fl)))
metrics.write("error(validation): %f" % (fl_val/(tr_val+fl_val)))

train_out.close()
validate_out.close()
metrics.close()