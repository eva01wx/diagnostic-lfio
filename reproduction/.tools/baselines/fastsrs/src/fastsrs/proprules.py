# Import generic packages
from __future__ import division
import pickle
import pandas as pd 
from math import isnan
import numpy as np
from math import lgamma
import itertools
from numpy.random import random
from random import sample, choices, seed,shuffle
import time
from copy import deepcopy
import operator
from collections import Counter, defaultdict
from fim import fpgrowth,fim 
from scipy.sparse import csc_matrix
from .util import *
from sklearn.ensemble import RandomForestClassifier
import math
from joblib import Parallel, delayed
import multiprocessing

# Definition of the class object ORS
class ORS(object):   
    
# Function to initialize ORS
# self is to make reference to ORS
# binary_data, Y: Xtrain, Ytrain

    def __init__(self, binary_data,Y,col_ls,method):
        self.attributeLevelNum = defaultdict(int) # 'attribute': number of levels
        self.attributeLevelNames = defaultdict(list) # 'attribute': indeces/names of levels
        self.attributeLevelIndeces = defaultdict(list) # 'attribute': list of indeces from the whole list 
        self.itemNames = defaultdict(list) # 'index from the whole list': attribute and level
        self.colindices = defaultdict(list) # 'attribute': index from the whole list
        self.attributeNames = [] # names of attributes without repetitions
        self.col_ls=col_ls
        self.df = binary_data # Xtrain
        self.Y = Y # Ytrain
        self.N = len(Y) # number of individuals
        if method=="fpgrowth":
            for i,name in enumerate(self.col_ls): # i: index, name: name of the variable
                attribute = name.split('_')[0]
                self.attributeNames.append(attribute)
                value = ''.join(name.split('_')[1:])
                self.attributeLevelNum[attribute] += 1
                self.attributeLevelNames[attribute].append(value)
                self.attributeLevelIndeces[attribute].append(i)
                self.itemNames[i] = name
                self.colindices[name] = i
        self.attributeNames = list(set(self.attributeNames))

# Function to define the parameters of MRS
# c1: penalization parameter for the number of literals
# c2: penalization parameter for the number of rules
    def set_parameters(self,c1=1,c2=1,c3=1):
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3

# Function to set the fixed bound found in Theorems 1,2,3
    def set_fixed_bounds(self):
        # Theorem 1:
        self.minnegsupp = np.sum(self.Y) - self.N*(self.c1 + self.c2 + self.c3)
        # Theorem 2:
        self.minpossupp = np.ceil((self.c1 + self.c2 + self.c3)*self.N)

# Function to premine rules with support supp and maximum length maxlen
# supp: minimum support of rules to be generated, typically a number in [5,15] (just for fpgrowth) (in %)
# maxlen: maximum length of a pattern to be generated, typically 3
# N: number of rules selected from all rules that are generated
# method: 'fpgrowth' or 'randomForest'
# criteria: 'IG' or 'precision'
# add_rules: list of rules to be added explicitly to those generated
########
### N = Nrules
### method = 'fpgrowth'
### criteria = 'IG'
### add_rules = []
########
    def generate_rules(self,supp,maxlen,N_rules,method = 'fpgrowth',criteria = 'IG',add_rules = []):
        np.random.seed(1)
        seed(1)
        self.maxlen = maxlen
        self.supp = supp
        if method =='fpgrowth':
            #print('Using fpgrowth to generate rules with support {} and max length {}'.format(supp,maxlen))            
            itemMatrix=[]
            for i in range(self.df.shape[0]):
                item_ls=[]
                for j in range(self.df.shape[1]):
                    if self.df[i][j]==1:
                        item_ls.append(self.col_ls[j])
                itemMatrix.append(item_ls)
            # itemMatrix is a list of N lists composed by the binary variables satisfies per individual
            pindex = np.where(self.Y==1)[0] # indices of positive examples
            start_time = time.time()
            rules=[]
            start_time = time.time()
            self.nrulesperlength = [] 
            self.lenindices = [] 
            ii = 0
            for z in range(1,maxlen+1):
                minpossupp = np.ceil((self.c1*z + self.c2 + self.c3*z)*self.N)
                rules_ls = fpgrowth([itemMatrix[i] for i in pindex],supp = 5,zmin = z,zmax = z)
                rules=rules+rules_ls
                hmr = len(rules_ls)
                self.nrulesperlength.append(hmr)
                self.lenindices.append([ii,ii + hmr])
                ii = ii + hmr
            rules = [np.sort(x[0]).tolist() for x in rules] # sort conditions in rules
            df = self.df
            col_ls=self.col_ls
        else:
            rules = []
            start_time = time.time()
            for length in [maxlen]:
                n_estimators = 500
                clf = RandomForestClassifier(n_estimators = n_estimators,max_depth = length, random_state=1)
                # specify RF with trees up to max_depth = maxlen
                clf.fit(self.df,self.Y)
                for n in range(n_estimators):
                    rules.extend(extract_rules(clf.estimators_[n],self.col_ls))
                    # extracting rules for each tree and each RF
            rules = [list(x) for x in set(tuple(np.sort(x)) for x in rules)] #sort rules
            df = 1-self.df  # computing the negative counterparts 
            df_columns = [name.strip() + '_neg' for name in self.col_ls] # call the negative counterparts
            df=np.concatenate((self.df, df), axis=1)
            col_ls=np.concatenate((self.col_ls, df_columns), axis=None)
            self.col_ls=col_ls
            self.df=df
            for i,name in enumerate(self.col_ls): # i: index, name: name of the variable
                attribute = name.split('_')[0]
                self.attributeNames.append(attribute)
                value = ''.join(name.split('_')[1:])
                self.attributeLevelNum[attribute] += 1
                self.attributeLevelNames[attribute].append(value)
                self.attributeLevelIndeces[attribute].append(i)
                self.itemNames[i] = name
                self.colindices[name] = i
        self.generate_time = time.time() - start_time
        index = []
        if len(rules) == 0:
            # Regularization too large: minpossupp exceeded the available
            # positive support, so the rule miner produced nothing. Set empty
            # state for all rule-count attributes that screen_rules would
            # otherwise populate (this variant has no train()).
            self.rulespace = [0]
            self.nrulesMETHOD = 0
            self.nrulesNEGSUPP = 0
            self.nrulesPOSSUPP = 0
            self.nrulesFINAL = 0
            self.all_rulelen = np.array([])
            self.rules = []
            self.RMatrix = np.zeros((self.N, 0), dtype=int)
            self.supp = np.array([])
            return
        self.rulespace = [len(rules)] # number of rules generated
        self.nrulesMETHOD = len(rules)
        self.all_rulelen = np.array([len(rule) for rule in rules]) # list of lengths of rules generated
        #print("Rules generated by fpgrowth", rules)
        n_cores = 1
        self.N_part=math.floor(N_rules/n_cores)
        rules=np.array(rules,dtype=object)
        rl=np.array_split(rules, n_cores)
        start_time=time.time()
        rule_ls, RMatrix_ls,supp_ls = zip(*Parallel(n_jobs=1)(delayed(self.screen_rules)(r) for r in rl))
        screen_time = time.time() - start_time
        #print('\tTook %0.3fs to screen rules' % (screen_time))
        self.rules = sum(rule_ls, [])
        self.RMatrix=np.hstack(RMatrix_ls)
        self.supp=np.hstack(supp_ls)

# Function to screen rules
# rules: set of rules generated in function generate_rules.
# df: dataset with (RF) or without (fpgrowth) negative counterparts 
# N: number of rules selected from all rules that are generated
# supp: minimum support of rules to be generated, typically a number in [5,15]
# criteria: 'IG' or 'precision'
# add_rules: list of rules to be added explicitly to those generated
    def screen_rules(self,rules): 
        method="fpgrowth"
        criteria = 'precision'
        itemInd = {} # 'attribute' (maybe with neg): index
        for i,name in enumerate(self.col_ls):
            itemInd[name] = int(i)
        len_rules = [len(rule) for rule in rules] # list of lengths of rules generated
        indices = np.array(list(itertools.chain.from_iterable([[itemInd[x] for x in rule] for rule in rules])))
        # concatenation of indices that satify the rules (in order)
        indptr = list(accumulate(len_rules)) # accumulated sum of the lengths of rules generated
        indptr.insert(0,0) # adding 0 as the first argument
        indptr = np.array(indptr) # indices for columns in ruleMatrix
        data = np.ones(len(indices)) # array of 1s of length the number of indices
        ruleMatrix = csc_matrix((data,indices,indptr),shape = (len(self.col_ls),len(rules)))
        mat = np.matrix(self.df) * ruleMatrix        
        # matrix (numberofindividuals)x(numberofrules): integer saying how much conditions in a rule j satisfies individual i
        lenMatrix = np.matrix([len_rules for i in range(self.df.shape[0])])
        # matrix (number of individuals)x(number of rules): repeated vector of len_rules in order to check if it is equal to mat or not
        Z =  (mat ==lenMatrix).astype(int) 
        # matrix (number of individuals)x(number of rules): 1 if individual i satisfies rule j
        Zpos = [Z[i] for i in np.where(self.Y>0)][0]
        # submatrix of Z with just the negative examples
        Zneg = [Z[i] for i in np.where(self.Y<1)][0]
        # submatrix of Z with just the negative examples
        TP = np.array(np.sum(Zpos,axis=0).tolist()[0])
        # vector (number of rules): int saying how many positive examples satisfy each rule (TP)
        TN = np.array(np.sum(Zneg,axis=0).tolist()[0])
        # vector (number of rules): int saying how many negative examples satisfy each rule (TP)        
        indT1 = np.where(TN<self.minnegsupp)[0]
        # Theorem 1:
        rulesT1 = [rules[i] for i in indT1]
        rules = rulesT1.copy()
        self.nrulesNEGSUPP = len(rules)
        ZT1 = Z[:,indT1] 
        Z = ZT1.copy()
        ZposT1 = Zpos[:,indT1]
        Zpos = ZposT1.copy()
        TPT1 = TP[indT1]
        TP = TPT1.copy()     
        nlen = []
        minpossupp = []
        for z in range(1,self.maxlen+1):
            minpossupp.append(int(np.ceil((self.c1*z + self.c2 + self.c3*z)*self.N)))
            how = np.sum(np.array([len(rule) for rule in rules])==z)
            nlen.append(how)      
        indz1 = range(0,nlen[0]) # rules with len 1
        indz2 = range(nlen[0],(nlen[0]+nlen[1])) # rules with len 2
        indRF = np.where(TP>minpossupp[0])[0] # rules with min supp len 1
        indRFbis = np.where(TP>minpossupp[1])[0] # rules with min supp len 2
        set1 = set(indz1)
        set2 = set(indRF)
        rulesRF = [rules[i] for i in list(set1.intersection(set2))]
        set1 = set(indz2)
        set2 = set(indRFbis)
        rulesRFbis = [rules[i] for i in list(set1.intersection(set2))]
        rulessss = rulesRF+rulesRFbis
        rules = rulessss.copy()
        self.nrulesPOSSUPP = len(rules)
        ZRF = Z[:,indRF]
        Z = ZRF.copy()
        ZposRF = Zpos[:,indRF]
        Zpos = ZposRF.copy()
        TPRF = TP[indRF]
        TP = TPRF.copy()
        supp_select = np.arange(0,len(rules))
        if len(supp_select)<=self.N_part: # if the number of rules generated is less than N, DONE, select all the rules generated.
            rules = [rules[i] for i in supp_select] # list of rules (self.rules = rules)
            RMatrix = np.array(Z[:,supp_select]) # matrix Z (self.RMatrix = Z)
            rules_len = [len(set([name.split('_')[0] for name in rule])) for rule in rules] # vector len_rules (self rules_len = len_rules)
            supp = np.array(np.sum(Z,axis=0).tolist()[0])[supp_select] 
            self.nrulesFINAL = self.nrulesPOSSUPP
        else: # if not, select N according to criteria
            self.nrulesFINAL = self.N_part
            FP = np.array(np.sum(Z,axis = 0))[0] - TP # FP: P - TP, per rule generated
            TN = len(self.Y) - np.sum(self.Y) - FP # TN: (N - pos) - FP, per rule generated
            FN = np.sum(self.Y) - TP
            p1 = TP.astype(float)/(TP+FP) # precision or true positive rate, per rule generated
            p2 = FN.astype(float)/(FN+TN) # false negative rate, per rule generated
            pp = (TP+FP).astype(float)/(TP+FP+TN+FN) # proportion of positives, per rule generated
            self.precision = p1
            if criteria =='precision':
                select = np.argsort(p1[supp_select])[::-1][:self.N_part].tolist() # indeces of the N rules to be chosen according better precision
            else:  
                cond_entropy = -pp*(p1*np.log(p1)+(1-p1)*np.log(1-p1))-(1-pp)*(p2*np.log(p2)+(1-p2)*np.log(1-p2))
                cond_entropy[p1*(1-p1)==0] = -((1-pp)*(p2*np.log(p2)+(1-p2)*np.log(1-p2)))[p1*(1-p1)==0]
                cond_entropy[p2*(1-p2)==0] = -(pp*(p1*np.log(p1)+(1-p1)*np.log(1-p1)))[p2*(1-p2)==0]
                cond_entropy[p1*(1-p1)*p2*(1-p2)==0] = 0
                pos = (TP+FN).astype(float)/(TP+FP+TN+FN) # pos = pp
                info = - pos * np.log(pos) - (1-pos)*np.log(1-pos)
                info[np.where((pos==1)| (pos==0))[0]] = 0
                IGR = (info - cond_entropy)/info 
                IGR[np.where(info==0)[0]] = 0
                select = np.argsort(IGR[supp_select])[::-1][:self.N_part].tolist() # indeces of the N rules to be chosen according better IG
            ind = list(supp_select[select]) # ind = select
            rules = [rules[i] for i in ind] # sublist of rules with N rules
            RMatrix = np.array(Z[:,ind]) # submatrix of Z, (number of individuals) x (N rules): 1 if individual i satisfies rule j
            score = p1[select] # save score precision just with the N rules to be used afterwards
            #TP = TP # save TP, per rule generated. WHY not just the N rules?
            rules_len = [len(set([name.split('_')[0] for name in rule])) for rule in rules] # rule_len for the N rules
            supp = np.array(np.sum(Z,axis=0).tolist()[0])[ind] 
       
        return rules, RMatrix,supp

      