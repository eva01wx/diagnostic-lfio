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

#from sklearn.metrics import confusion_matrix

# Definition of the class object MRS
class ORS(object):   
    
# Function to initialize MRS
# self is to make reference to MRS
# binary_data, Y: Xtrain, Ytrain
#####
# self = model
# binary_data = Xtrain
# Y = Ytrain
#####
    #def __init__(self, binary_data,Y,col_ls,dataframe):
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
        ### NEW PART 
        pos_s=np.where(Y == 1)[0] 
        neg_s=np.where(Y == 0)[0]     
        prop_select=0.25 #original
        #prop_select= 0.5 # 2
        n_sampled=math.floor(prop_select*len(pos_s))
        self.pos_idx=np.random.choice(pos_s, size=math.floor(prop_select*len(pos_s)), replace=False) 
        self.neg_idx=np.random.choice(neg_s,size=math.floor(prop_select*len(neg_s)),replace=False)  
        #self.dataframe=dataframe
        ####
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
        ### NEW PART 
        df_pos=self.df[self.pos_idx,:]
        Y_pos=self.Y[self.pos_idx]
        df_neg=self.df[self.neg_idx,:]
        Y_neg=self.Y[self.neg_idx]
        self.df_part=np.concatenate((df_pos, df_neg), axis=0)
        self.Y_part=np.concatenate((Y_pos, Y_neg), axis=0)
        ### 
# Function to define the parameters of MRS
# c1: penalization parameter for the number of literals
# c2: penalization parameter for the number of rules
    def set_parameters(self,c1=1,c2=1,c3=1):
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        ###NEW
        #self.iters_addrule = []
        #self.iters_rmrule = []
        #self.iters_addcondition = [] 
        #self.iters_rmcondition = []
        #self.iters_addvalue = []
        #self.iters_replacerule = [] 
        #self.times_addrule = []
        #self.times_rmrule = []
        #self.times_addcondition = [] 
        #self.times_rmcondition = []
        #self.times_addvalue = []
        #self.times_replacerule = [] 
        ###

# Function to set the fixed bound found in Theorems 1,2,3
    def set_fixed_bounds(self):
        # Theorem 1:
        self.minnegsupp = np.sum(self.Y) - self.N*(self.c1 + self.c2 + self.c3)
        # Theorem 2 (conservative static bound, assumes len=val=1):
        self.minpossupp = np.ceil((self.c1 + self.c2 + self.c3)*self.N)
        # Theorem 3:
        #self.staticmaxrulesetsize = np.sum(self.Y)/(self.N*(self.c1 + self.c2 + self.c3))
        # Theorem 4: pt_curr/(model.c1+model.c2+model.c3)
#        self.dynamicmaxrulesetsize

# Theorem 2: supp^S+(r) >= len(r)*C1 + C2 + val(r)*C3
    def _minpossupp(self, rule_len, rule_val):
        return np.ceil((self.c1 * rule_len + self.c2 + self.c3 * rule_val) * self.N)

# Wrapper for Theorem 2 given a rule dict; extra_len/extra_val account for
# the condition/value being considered for addition before it is committed.
    def _minpossupp_rule(self, subdict, extra_len=0, extra_val=0):
        rule_len = len(subdict) + extra_len
        rule_val = sum(len(v) for v in subdict.values()) + extra_val
        return self._minpossupp(rule_len, rule_val)

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
        seed(1)
        np.random.seed(1)
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
            #itemMatrix = [[item for item in self.dataframe.columns if row[item] ==1] for i,row in self.dataframe.iterrows() ]  
            #print("ItemMatrix",itemMatrix)
            #print("ItemMatrix1",itemMatrix_1)
            # itemMatrix is a list of N lists composed by the binary variables satisfies per individual
            # itemMatrix takes long to be computed
            pindex = np.where(self.Y==1)[0] # indices of positive examples
            ###NEW
            #self.trainingerrortrivialmodel = round(1-len(pindex)/self.N,3)
            ###
            #nindex = np.where(self.Y!=1)[0] # indices of negative examples
            start_time = time.time()
#            rules = fpgrowth([itemMatrix[i] for i in pindex],supp = supp,zmin = 1,zmax = maxlen)
#            # fpgrowth generates rules of maximum length zmax and with positive support supp
            #print(100*self.minpossupp/len(pindex))
            #print(self.minpossupp)
            #Sup = np.max(np.asarray([5,100*self.minpossupp/len(pindex)]))
            #Sup = 5
            #rules = fpgrowth([itemMatrix[i] for i in pindex],supp = Sup,zmin = 1,zmax = maxlen)
            #rules = fpgrowth([itemMatrix[i] for i in pindex],supp = Sup,zmin = 3,zmax = maxlen)
            #rules = fpgrowth([itemMatrix[i] for i in pindex],supp = 15,zmin = 1,zmax = maxlen)
            rules=[]
            start_time = time.time()
            ###NEW
            self.nrulesperlength = [] # [10,20,50] Total = 80
            self.lenindices = [] # [[0,10],[10,30],[30,80]]
            ###
            ii = 0
            for z in range(1,maxlen+1):
                # Theorem 2 with len(r)=val(r)=z: FP-growth produces binary
                # single-value items, so each condition contributes exactly one value.
                minpossupp = self._minpossupp(z, z)
                rules_ls = fpgrowth([itemMatrix[i] for i in pindex],supp = np.max(np.asarray([5,100*minpossupp/len(pindex)])),zmin = z,zmax = z)
                #print("z size",z)
                rules=rules+rules_ls
                ###NEW
                hmr = len(rules_ls)
                self.nrulesperlength.append(hmr)
                self.lenindices.append([ii,ii + hmr])
                ii = ii + hmr
                ###
            ###NEW
            self.distributionrules = np.zeros(len(rules)) +  0.001
            ###
            # fpgrowth generates rules of maximum length zmax and with positive support supp
            rules = [np.sort(x[0]).tolist() for x in rules] # sort conditions in rules
            df = self.df
            col_ls=self.col_ls
            #print("fogrowth finished")
            #print('fpgrowth',rules[0:10])
        else:
            #print('Using random forest to generate rules ...')
            rules = []
            start_time = time.time()
        #    for length in range(2,maxlen+1,1):
            for length in [maxlen]:
                #n_estimators = 500*length# min(5000,int(min(comb(df.shape[1], length, exact=True),10000/maxlen)))
                n_estimators = 500
                clf = RandomForestClassifier(n_estimators = n_estimators,max_depth = length, random_state=1)
                # specify RF with trees up to max_depth = maxlen
                #print('RF', self.df.shape[1]/2)
                #clf.fit(self.df[:,list(range(int(self.df.shape[1]/2)))],self.Y)
                clf.fit(self.df,self.Y)
                for n in range(n_estimators):
                    #rules.extend(extract_rules(clf.estimators_[n],self.col_ls[:int(self.df.shape[1]/2)]))
                    rules.extend(extract_rules(clf.estimators_[n],self.col_ls))
                    # extracting rules for each tree and each RF
            rules = [list(x) for x in set(tuple(np.sort(x)) for x in rules)] #sort rules
            #print('RF',rules[0:20])
            #print('length',len(rules))
            df = 1-self.df  # computing the negative counterparts 
            #print('self.col_ls',self.col_ls[0:20])
            df_columns = [name.strip() + '_neg' for name in self.col_ls] # call the negative counterparts
            #print('df_columns',df_columns[0:20])
            df=np.concatenate((self.df, df), axis=1)
            #df = pd.concat([self.df,df],axis = 1) # create a data matrix of double size including the negative counterparts
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
        # print("Print time main--------------------------------------------")
        # print('\tTook %0.3fs to generate %d rules' % (self.generate_time, len(rules)))
        # print("-----------------------------------------------------------")
        # The following is in case that add_rules is not empty
        index = []
        ###CMR this can be commented
#        for rule in add_rules:
#            if np.sort(rule).tolist()  not in rules:
#                rules.append(rule)
#                index.append(len(rules)-1)
#            else:
#                index.append(rules.index(rule))
        ###
        if len(rules) == 0:
            # Regularization too large: minpossupp exceeded the available
            # positive support, so the rule miner produced nothing. Skip the
            # parallel screen; train() will short-circuit to the empty rule
            # set with a warning.
            self.rulespace = [0]
            self.all_rulelen = np.array([])
            self.rules = []
            self.RMatrix = np.zeros((self.N, 0), dtype=int)
            self.supp = np.array([])
            return
        self.rulespace = [len(rules)] # number of rules generated
        self.all_rulelen = np.array([len(rule) for rule in rules]) # list of lengths of rules generated
        #print("Rules generated by fpgrowth", rules)
        #self.screen_rules(rules) # select the top N rules using secondary criteria
        n_cores=multiprocessing.cpu_count()
        self.N_part=math.floor(N_rules/n_cores)
        #rl = [rules[10*i:10*(i+1)] for i in range(math.floor(len(rules)/10))]
        #print("rules before shuffle",rules)
        shuffle(rules)
        #print("rules", rules)
        rules=np.array(rules,dtype=object)
        rl=np.array_split(rules, n_cores)
        #print("rl[0]",rl[0])
        #print(len(rl))
        start_time=time.time()
        rule_ls, RMatrix_ls,supp_ls,RMatrix_part_ls = zip(*Parallel(n_jobs=-1)(delayed(self.screen_rules)(r) for r in rl))
        screen_time = time.time() - start_time
        #print('\tTook %0.3fs to screen rules' % (screen_time))
        self.rules = sum(rule_ls, [])
        self.RMatrix=np.hstack(RMatrix_ls)
        #print(len(RMatrix_ls))
        #print("shape RMAtrix",self.RMatrix.shape)
        self.supp=np.hstack(supp_ls)
        self.RMatrix_part=np.hstack(RMatrix_part_ls)
        #print("rules selected after screening",len(self.rules) )
        #print("shape RMAtrix_part",self.RMatrix_part.shape)
        #print("finished generating rules")
# Function to screen rules
# rules: set of rules generated in function generate_rules.
# df: dataset with (RF) or without (fpgrowth) negative counterparts 
# N: number of rules selected from all rules that are generated
# supp: minimum support of rules to be generated, typically a number in [5,15]
# criteria: 'IG' or 'precision'
# add_rules: list of rules to be added explicitly to those generated
    def screen_rules(self,rules):
        seed(1)
        np.random.seed(1)
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
        # matrix (number of variables)x(number of rules), 1 if variable i appears at rule j
        #print('size df',df.shape)
        #print('size ruleMatrix',ruleMatrix.shape)
       # print(np.matrix(df).shape)
        #print(ruleMatrix.shape)
        mat = np.matrix(self.df) * ruleMatrix 
        
        # matrix (numberofindividuals)x(numberofrules): integer saying how much conditions in a rule j satisfies individual i
        # mat takes long to be computed
        lenMatrix = np.matrix([len_rules for i in range(self.df.shape[0])])
        # matrix (number of individuals)x(number of rules): repeated vector of len_rules in order to check if it is equal to mat or not
        # lenMatrix takes long to be computed
        Z =  (mat ==lenMatrix).astype(int) 
        # matrix (number of individuals)x(number of rules): 1 if individual i satisfies rule j
        # Z takes long to be computed
        Zpos = [Z[i] for i in np.where(self.Y>0)][0]
        # submatrix of Z with just the negative examples
        Zneg = [Z[i] for i in np.where(self.Y<1)][0]
        # submatrix of Z with just the negative examples
        TP = np.array(np.sum(Zpos,axis=0).tolist()[0])
        # vector (number of rules): int saying how many positive examples satisfies each rule (TP)
        TN = np.array(np.sum(Zneg,axis=0).tolist()[0])
        # vector (number of rules): int saying how many negative examples satisfies each rule (TP)        
        indT1 = np.where(TN<self.minnegsupp)[0]
        # Theorem 1:
        rulesT1 = [rules[i] for i in indT1]
        rules = rulesT1.copy()
        ZT1 = Z[:,indT1] 
        Z = ZT1.copy()
        ZposT1 = Zpos[:,indT1]
        Zpos = ZposT1.copy()
        TPT1 = TP[indT1]
        TP = TPT1.copy()     
        if method=='randomForest': #in the screening phase RF was not generated with a minimum support
            indRF = np.where(TP>self.minpossupp)[0]
            rulesRF = [rules[i] for i in indRF]
            rules = rulesRF.copy()
            ZRF = Z[:,indRF]
            Z = ZRF.copy()
            ZposRF = Zpos[:,indRF]
            Zpos = ZposRF.copy()
            TPRF = TP[indRF]
            TP = TPRF.copy()
        #supp_select = np.array([i for i in range(len(TP))]) # np.arange(0,len(TP))
        supp_select = np.arange(0,len(TP))
        if len(supp_select)<=self.N_part: # if the number of rules generated is less than N, DONE, select all the rules generated.
            rules = [rules[i] for i in supp_select] # list of rules (self.rules = rules)
            RMatrix = np.array(Z[:,supp_select]) # matrix Z (self.RMatrix = Z)
            rules_len = [len(set([name.split('_')[0] for name in rule])) for rule in rules] # vector len_rules (self rules_len = len_rules)
            #print('name.split',name.split('_')[0])
            supp = np.array(np.sum(Z,axis=0).tolist()[0])[supp_select] 
            RMatrix_pos=RMatrix[self.pos_idx,:]
            RMatrix_neg=RMatrix[self.neg_idx,:]
            RMatrix_part=np.concatenate((RMatrix_pos, RMatrix_neg), axis=0)
            #return rules, RMatrix,rules_len,supp,RMatrix_part
            # select supp indeces in the vector composed by the number of pos and neg examples satisfiying the rules 
        else: # if not, select N according to criteria
            FP = np.array(np.sum(Z,axis = 0))[0] - TP # FP: P - TP, per rule generated
            TN = len(self.Y) - np.sum(self.Y) - FP # TN: (N - pos) - FP, per rule generated
            FN = np.sum(self.Y) - TP # FN: positive uncorrectly classified, per rule generated
            #print("TP",TP)
            #print("FP",FP)
            #print("TN",TN)
            #print("FN",FN)
            p1 = TP.astype(float)/(TP+FP) # precision or true positive rate, per rule generated
            p2 = FN.astype(float)/(FN+TN) # false negative rate, per rule generated
            pp = (TP+FP).astype(float)/(TP+FP+TN+FN) # proportion of positives, per rule generated
            self.precision = p1
            if criteria =='precision':
                select = np.argsort(p1[supp_select])[::-1][:self.N_part].tolist() # indeces of the N rules to be chosen according better precision
            else: # 
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
            # RMatrix takes a bit
            score = p1[select] # save score precision just with the N rules to be used afterwards
            rules_len = [len(set([name.split('_')[0] for name in rule])) for rule in rules] # rule_len for the N rules
            supp = np.array(np.sum(Z,axis=0).tolist()[0])[ind] # supp for the Nrules
        ### NEW PART 
            RMatrix_pos=RMatrix[self.pos_idx,:]
            RMatrix_neg=RMatrix[self.neg_idx,:]
            RMatrix_part=np.concatenate((RMatrix_pos, RMatrix_neg), axis=0)
       
        return rules, RMatrix,supp,RMatrix_part
        ### 

        #self.screen_time = time.time() - start_time
       #print('\tTook %0.3fs to screen %d rules' % (self.screen_time, len(self.rules)))
        #print('\tTook %0.3fs to screen and obtain %d rules' % (self.screen_time, len(self.rules)))

# Function to initialize parameters
    def precompute(self):
        #self.dynamicmaxrulesetsize = [self.staticmaxrulesetsize] 
        return

# Function to train MRS using SA
# Niteration: number of iterations in each chain in SA
# q: probability of being random
# print_message: show some messages (True) or not (False)
# init: initial rule set
#######
### print_message = True
### init = []
#######
    def train(self, Niteration = 2000, q=0.3, print_message=True,init = []):
        #bound_count=0
        #self.precompute()
        start_time = time.time()
        self.Y = np.array(self.Y)
        ### NEW PART 
        self.Y_part=np.array(self.Y_part)
        ### 
        """ parameters for Simulated Annealing """
        #T0 = 10000 # initial temperature
        T0=20
        #TN = 0.001
        pt_max = 100000000 # initial objective
        self.maps=[] # list of ?
        self.objvalues = []
        if len(self.rules) == 0:
            # Regularization too large: no rule survived screening, so the
            # empty rule set (predicts class 0 for every example) is optimal.
            import warnings
            warnings.warn(
                "Regularization (c1+c2+c3) is too large: no rule satisfies the "
                "minimum negative-support bound. Returning an empty rule set, "
                "which predicts class 0 (negative) for every example.",
                stacklevel=2,
            )
            return [], self.maps
        if init == []: # if no rules were generated?
            seed(1)
            np.random.seed(1)
            N = sample(range(1,8,1),1)[0] # random number between 1 and 7. I guess random number of rules.
            ###NEW
            #print("------->len(self.rules)<-------",len(self.rules))
            rulesindices = sample(range(1,len(self.rules)),N)

            MRS_curr=[] # list storing the current rule set 
            for n in range(N): # for each rule, generate conditions
                subdict = defaultdict(list) # dictionary for each of the rules for MRS_curr
                rule = self.rules[rulesindices[n]]
                for condition in rule:
                    Str_list=condition.split('_')
                    att=Str_list[0]
                    subdict[att]= [self.colindices[condition]] # add this condition to the dictionary together with values
                MRS_curr.append(subdict) # adding rules to MRS_curr as dictionaries
            ###
        else:
            MRS_curr = init # if initial rule list is directly given
        self.MRS_curr = MRS_curr # save MRS_curr in model
        #######
        ### MRS_new = MRS_curr
        #######
        #####
        ### NEW PART 
        MRS_curr,_,prob,Yhat_curr,W_curr,V_curr,MRS_curr_rlens,MRS_curr_len = self.compute_prob(MRS_curr,self.df_part,self.Y_part) # compute current solution accuracy ???????
        ### 
        pt_curr = prob[0] + self.c1*prob[1] + self.c2*prob[2] + self.c3*prob[3] # current objective value
        self.dynamicmaxrulesetsize = [pt_curr/(self.c1+self.c2+self.c3)]
        #self.memory = {} # initialize set of models to store in memory
        self.objvalues = []
        #stopping_value = 501
        #stopping_value = 30 #####2 
        ###CMR
        ### NEW PART
        ntimes_mis = np.zeros(len(self.Y_part)) 
        ###
        ### 
        for iter in range(Niteration): 
            #print('iter',iter) 
            #######
            ### iter = 0
            #######
         #   if print_message:
         #       print('========= iteration = {} ========='.format(iter))
            #if iter>0.75*Niteration: # if iteration is at least the number 75% . change to 90% also higher tempertaure--dont have to converge
            ### NEW PART 
            #if iter<math.floor(0.5*Niteration): # original
            if iter<math.floor(0.25*Niteration): #3
                df=self.df_part
                Y=self.Y_part
                RMatrix=self.RMatrix_part
            else:
                df=self.df
                Y=self.Y
                RMatrix=self.RMatrix
            #print('int(0.5*Niteration)', int(0.5*Niteration))
            #if iter == math.floor(0.5*Niteration): # original
            if iter == math.floor(0.25*Niteration): #3
                MRS_curr,_,prob,Yhat_curr,W_curr,V_curr,MRS_curr_rlens,MRS_curr_len = self.compute_prob(MRS_curr,self.df,self.Y)
                ntimes_mis=np.zeros(self.N)
            ###  
            # if iter>0.9*Niteration:
            #     Yhat_curr,pt_curr,W_curr,V_curr,MRS_curr_rlens,MRS_curr_len = Yhat_max.copy(),pt_max,W_max.copy(),V_max.copy(),MRS_max_rlens.copy(),MRS_max_len
            #     MRS_curr = [subdict for subdict in MRS_max]
            MRS_new = [deepcopy(subdict) for subdict in MRS_curr] # save MRS_curr in MRS_new
            #######
            ### MRS = MRS_new
            #######
            if print_message:
               # print('before action, the new set is:')
                self.printMRS(MRS_new)
            added_rule = False # initialize value to False if no rule has been added to rule list
            #self.memory[iter] = MRS_new # 1 model per iteration
            _,_,temp,_,_,_,_,_ = self.compute_prob(MRS_new,df,Y)
            self.objvalues.append(temp[0] + self.c1*temp[1] + self.c2*temp[2] + self.c3*temp[3])
            #print('objvalues',self.objvalues)
            #if iter > stopping_value:  
            ### NEW PART    
            # if iter > 0.5*Niteration + stopping_value: #####
            # ###
            # #if iter>0.75*Niteration:
            #     stopping_list = self.objvalues[(iter-stopping_value):iter] #####
            #     if all(x==stopping_list[0] for x in stopping_list):#####
            #         #print('stoppingcriterion')
            #         break #####
            # if rule set has 1 single rule or the random value is over certain value (?)
#            if random()>np.exp(-np.true_divide(iter,1.5*Niteration)) or MRS_curr_len ==1: # proposing a new solution (?)
            logical_incorr = Y!=Yhat_curr
            # print("iter",iter)
            # print("logical_incorr",logical_incorr)
            # print("Y", len(Y))
            # print("Yhat_curr", len(Yhat_curr))
            #print('logical_incorr',logical_incorr)
            incorr = np.where(logical_incorr)[0]
            #print('incorr',incorr)
            #incorrnot0 = np.where(logical_incorr) # indices of misclassified examples
            #print('incorrnot0',incorrnot0)
            if len(incorr)==0: # every data point is correctly classified
                trim = False # initialize trim to False (?)
                ### NEW PART
                #if iter < 0.5*Niteration: #original
                #    iter = 0.5*Niteration #original
                if iter < 0.25*Niteration: #3
                    iter = 0.25*Niteration #3
                else:
                    break # finish
                ####
                # it means the MRS correctly classified all points but the rule could be redundant, so triming is needed
            else: # some example is misclassified
                trim = False
                ###CMR
                #print('ntimes_mis_before',ntimes_mis)
                #n_old=ntimes_mis
                #print('where missed',logical_incorr.astype(int))
                ntimes_mis = np.add(ntimes_mis,logical_incorr.astype(int))
                #print("len ntimes_mis",len(ntimes_mis))
                #print("len logical_incorr",len(logical_incorr))
                #print('iter',iter)
                #print('ntimes_mis',ntimes_mis
                #break
                #ntimes_new=ntimes_mis
                #print('ntimes_mis_after_diff',ntimes_new-n_old)
                #print('where missed',logical_incorr.astype(int))
                #print('list(incorr)',list(incorr))
                #print('ntimes_mis',ntimes_mis)
                #print('logical_incorr',logical_incorr)
                #print('ntimes_mis[logical_incorr]',ntimes_mis[logical_incorr])
                ### NEW PART
                if (iter > 30 and iter<0.5*Niteration) or (iter>0.5*Niteration+30): #original
                #if (iter > 30 and iter<0.25*Niteration) or (iter>0.25*Niteration+30): 
                ###
                    misc=ntimes_mis[logical_incorr]/(iter+1)
                    #print("ntimes_mis",ntimes_mis)
                    #print("logical_error",logical_incorr)
                    #print("ntimes_mis",ntimes_mis[logical_incorr])
                    #print("iter",iter)
                    #print("mis rate",misc)
                    #wei = iter+1 - ntimes_mis[logical_incorr]
                    wei=0.5-np.abs(0.5-misc)+1e-7
                    #ex = choices(list(incorr), weights=wei, k=1)[0]
                    c = choices(range(len(incorr)), weights=wei, k=1)[0]
                    ex=incorr[c]
                else:
                    ex = choices(list(incorr), k=1)[0]
                ex_dict = defaultdict(list) # dictionary with conditions that ex satisfies
                for i,name in enumerate(self.col_ls):
                    attr_name = name.split('_')[0]
                    if df[ex,i]==1:
                        ex_dict[attr_name].append(i)
             #   if print_message:
             #       print('ex={}'.format(ex))
            # randomly perturb if we have more than 1 rule in the rule set
            if random()<0 and MRS_curr_len>1:
                #if print_message:
                #    print('perturb')
                self.action_perturb(MRS_new,W_curr)
            else:
                if trim == False and random()<=0: # jump out of local minimum:
                    self.action_jump(MRS_new,W_curr,q,print_message,df,Y)
                elif trim==False and Y[ex]==1: # if the misclassified example is positive 
                    # it means the model failed to cover it, so we need to do one of three following
                    t = random()
                    # Deactivate the option of adding a rule if the maximum rule set size is overcome
                    if MRS_curr_len >= self.dynamicmaxrulesetsize[-1]:
                    #if MRS_curr_len >= min(self.staticmaxrulesetsize,self.dynamicmaxrulesetsize[-1]):
                        #bound_count=bound_count+1
                        #t = 0.667*t
                        t = (2/3)*t
                    # 1) add value to cover ex with probability 1/3
                    if t<(1/3): 
                        # find out which of the rules does ex satisfy all but one condition
                        temp  = [i for i in range(MRS_curr_len) if V_curr[i][ex] == MRS_curr_rlens[i]-1]
                        #######
                        ### Yhat = Yhat_curr
                        ### indices = temp
                        #######
                        ###NEW
                        #self.iters_addvalue.append(iter)
                        #timeaction1 = time.time()
                        self.action_addvalue(MRS_new,ex_dict,Yhat_curr,temp,q,print_message,df,Y)
                        #timeaction2 = time.time()
                        #self.times_addvalue.append(timeaction2-timeaction1)
                        ###
                    # 2) remove condition (from a rule) that ex does not satisfy with probability 1/3
                    elif t<(2/3) and not (MRS_curr_len==1 and len(MRS_new[0])==1) and MRS_curr_len>0: # do in case that we have at least 1 rule with 2 conditions
                        #######
                        ### V = V_curr
                        ### W = W_curr
                        #######
                        ###NEW
                        #self.iters_rmcondition.append(iter) 
                        #timeaction1 = time.time()
                        self.action_rmcondition(MRS_new,V_curr,W_curr,ex_dict,print_message,df,Y) # problematic to do that, DO NOT REMOVE CONDITION
                        #timeaction2 = time.time()
                        #self.times_rmcondition.append(timeaction2-timeaction1)
                        ###
                # 3) add a rule that ex satisfies
                    else: 
                        #######
                        ### Yhat = Yhat_curr
                        ### W = W_curr
                        #######
                        ###NEW
                        #self.iters_addrule.append(iter) 
                        #timeaction1 = time.time()
                        self.action_addrule(MRS_new,W_curr,Yhat_curr,ex,ex_dict,self.maxlen,q,print_message,iter,Y,RMatrix)
                        #timeaction2 = time.time()
                        #self.times_addrule.append(timeaction2-timeaction1)
                        added_rule = True
                        ###
                else: # the misclassified example is negative
                    # it means the model covers the wrong data, so we need to do one of the following 
                    t = random()
                    # 1) Add a condition to a rule that satisfies the example that DOES NOT satisfies such condition
                    if trim==False and (MRS_curr_len<=1 or t<=1/3):#or (random()<=1.0/(2-float(iter)/Niteration) and MRS_curr_len<=10: 
                        #######
                        ### W = W_curr
                        #######
                        ###NEW
                        #self.iters_addcondition.append(iter) 
                        #timeaction1 = time.time()
                        self.action_addcondition(MRS_new,ex_dict,W_curr,print_message,df,Y)
                        #timeaction2 = time.time()
                        #self.times_addcondition.append(timeaction2-timeaction1)
                        ###
                    # 2) Remove a rule satisfied by the misclassified example
                    elif t<= (2/3): 
                        #######
                        ### W = W_curr
                        #######
                        ###NEW
                        #self.iters_rmrule.append(iter) 
                        #timeaction1 = time.time()
                        self.action_rmrule(MRS_new,W_curr,print_message,q,Y)
                        #timeaction2 = time.time()
                        #self.times_rmrule.append(timeaction2-timeaction1)
                        ###
                    # 3) Replace a rule: remove + add
                    else: 
                        ###NEW
                        #self.iters_replacerule.append(iter) 
                        #timeaction1 = time.time()
                        self.action_rmrule(MRS_new,W_curr,print_message,q,Y) # remove a rule
                        MRS_new,cfmatrix,prob,Yhat_new,W_new,V_new,MRS_new_rlens,MRS_new_len = self.compute_prob(MRS_new,df,Y) # compute current solution accuracy
                        #print("Yhat_new", Yhat_new)
                        incorr = np.where(np.logical_and(Y==1, Yhat_new==0))[0] # indices of misclassified examples
                        if len(incorr)>0:
                            ex = sample(list(incorr),1)[0] # a random misclassified example
                            ex_dict = defaultdict(list) # dictionary with conditions that ex satisfies
                            for i,name in enumerate(self.col_ls):
                                attr_name = name.split('_')[0]
                                if df[ex,i]==1:
                                    ex_dict[attr_name].append(i)
                            self.action_addrule(MRS_new,W_new,Yhat_new,ex,ex_dict,self.maxlen,q,print_message,iter,Y,RMatrix) # add a rule
                            added_rule = True # save if a rule has been added to the model
                        #timeaction2 = time.time()
                        #self.times_replacerule.append(timeaction2-timeaction1)
            MRS_new,cfmatrix,prob,Yhat_new,W_new,V_new,MRS_new_rlens,MRS_new_len= self.compute_prob(MRS_new,df,Y) # compute current solution accuracy
#            else:
#                self.action_trymerge(MRS_new)
#            MRS_new,cfmatrix,prob,Yhat_new,W_new,V_new,MRS_new_rlens,MRS_new_len= self.compute_prob(MRS_new) # compute current solution accuracy
            if random()<0.25 and added_rule == True:
                #######
                ### W = W_new
                #######
                self.refine_rule(MRS_new,W_new,print_message,df,Y) # add a condition to a rule
                MRS_new,cfmatrix,prob,Yhat_new,W_new,V_new,MRS_new_rlens,MRS_new_len= self.compute_prob(MRS_new,df,Y) # compute current solution accuracy
            pt_new = prob[0] + self.c1*prob[1] + self.c2*prob[2] + self.c3*prob[3] # new objective value
            T = np.true_divide(T0, 1 + iter) # decreasing temperature
            #T = TN + (T0-TN)*np.true_divide(1,1+np.exp((iter-0.5*Niteration)*2*np.log(T0-TN)/Niteration))
            #print('T',T)
            #T = T0**(1 - iter/500)
            #print('ptcurr-ptnew', pt_curr-pt_new)
            alpha = np.exp(float(pt_curr -pt_new)/T) # probability of accepting new solution though worse
            #print('alpha',alpha)
            #if print_message: # print information of the current model and the new one
            #    print('original MRS is')
            #    self.printMRS(MRS_curr)
            #    print('new MRS is:')
            #    self.printMRS(MRS_new)
            #    TP,FP,TN,FN = cfmatrix
            #    tpr = float(TP)/(TP+FN)
            #    fpr = float(FP)/(FP+TN)
            #    print('accuracy = {}, tpr = {}, fpr = {}, TP = {},FN = {}, FP = {}, TN = {}\n pt_new is {}, Error= {}, NumOfLiterals = {}, NumOfRules = {}, NumOfTerms = {}\n'.format(float(TP+TN)/self.N,tpr,fpr,TP,FN,FP,TN,pt_new, prob[0], prob[1],prob[2],prob[3]))
            if pt_new < pt_max: # if the new solution generated is betten that the current one, update it by the new one in MRS_max
                TP,FP,TN,FN = cfmatrix
                tpr = float(TP)/(TP+FN)
                fpr = float(FP)/(FP+TN)
                ### NEW PART
                #if iter>=0.5*Niteration: #original
                if iter>=0.25*Niteration: #3
                    MRS_max = [deepcopy(subdict) for subdict in MRS_new]
                    pt_max,Yhat_max, W_max,V_max, MRS_max_rlens, MRS_max_len = pt_new, Yhat_new.copy(),W_new.copy(),V_new.copy(),MRS_new_rlens.copy(),MRS_new_len # save parameters of model MRS_max
                    self.MRS_max = MRS_max # save MRS_max into the model
                ###
                #self.dynamicmaxrulesetsize.append((pt_new - self.c1 - self.c3)/self.c2) #Theorem3 
                ###NEW
                #self.dynamicmaxrulesetsize.append((pt_new-self.trainingerrortrivialmodel)/(self.c1+self.c2+self.c3)) #Theorem4
                ###
                self.dynamicmaxrulesetsize.append(pt_new/(self.c1+self.c2+self.c3)) #Theorem4
                # self.dynamicmaxrulesetsize.append(pt_new/(self.c1+self.c2+self.c3)) #Theorem4
                # (pt_new - 0.3)/...
                #if print_message:
                #    print('\n** max at iter = {} ** \nTP = {},FN = {}, FP = {}, TN = {}\n pt_new is {}, Error= {}, NumOfLiterals = {}, NumOfRules = {}, NumOfTerms = {}\n accuracy = {}, tpr = {}, fpr = {}\n M = {}'.format(iter,TP,FN,FP,TN,pt_new, prob[0], prob[1],prob[2],prob[3],float(TP+TN)/(FP+TP+TN+FN),tpr,fpr,self.dynamicmaxrulesetsize[-1]))
                #    print('merged MRS_merged is')
                #    self.printMRS(MRS_max)
                    #print([sum(W_max[i]) for i in range(len(W_max))]) # list with the number of individuals that satisfies each rule
                #self.maps.append([iter,pt_new,MRS_new,cfmatrix, self.dynamicmaxrulesetsize[-1],time.time() - start_time])
            #print('alpha ',alpha)
            number = random()
            #print('number',number)
            #print('alpha',alpha)
            if number <= alpha: # if a random number is less or equal than alpha, we accept the new model though it is not that good
                
                #print('go into ',iter)
                MRS_curr = [deepcopy(subdict) for subdict in MRS_new]
                Yhat_curr,pt_curr,W_curr,V_curr,MRS_curr_rlens,MRS_curr_len = Yhat_new.copy(),pt_new,W_new.copy(),V_new.copy(),MRS_new_rlens.copy(),MRS_new_len
                #if print_message:
                #    print('ACCEPT the new MRS\n')
        MRS_merged = self.merge(MRS_max) # merge model MRS_max
        #print("\tBound Used ---------------",bound_count)
        #print('MRS_merged', MRS_merged)
        return MRS_merged, self.maps # return the new model together with some information stored in maps                 

# Function to add a condition to a rule in the model
    def refine_rule(self,MRS_new,W,print_message,df,Y):
        #if print_message:
        #    print('Select the BEST condition to add')
        score_max = 0 # initialize/lower bound of score
        add_i = -1 # initialize number of value to be added (last one) (?)
        for i,subdict in enumerate(MRS_new): # per rule in the model
#            if (sum(W[i,np.where(self.Y==1)[0]])>self.minpossupp) and (sum(W[i,np.where(self.Y==0)[0]])<self.minnegsupp): # checking if the rule satisfies the minimum required in Theorems
            # It is always verified because of the way we premine rules
            candidate_attributes =[i for i in self.attributeNames if i not in subdict.keys()] # attributes that are not in the rule
            for attr in candidate_attributes: # per attribute in the candidate list
                candidate_values = self.attributeLevelIndeces[attr] # values that belongs to the attribute
                for value in candidate_values: # per value in the candidate list
                    #TP,FP,TN,FN = getConfusion(W[i],Y) # accuracy current model
                    #TN, FP, FN, TP = confusion_matrix(self.Y,W[i]).ravel()
                    #score1 = float(TP)/(TP+FP+1) # score current model
                    score1 =np.dot(W[i],Y)/(np.sum(W[i])+1)
                    Z = W[i]*df[:,value] # W proposed model including this new value
                    TP,FP,TN,FN = getConfusion(Z,Y) # accuracy proposed model
                    #TN, FP, FN, TP = confusion_matrix(self.Y,Z).ravel()
                    #score2 = float(TP)/(TP+FP+1) # score proposed model
                    score2 =np.dot(Z,Y)/(np.sum(Z)+1)
                    score_d = score2 - score1 # score difference between proposed and current models                       
                    # update the best value to include according to score
                    rule_minpossupp = self._minpossupp_rule(subdict, extra_len=1, extra_val=1)
                    if score_d>=score_max and (TP>rule_minpossupp and FP<self.minnegsupp):
                        score_max = score_d
                        add_attribute = attr
                        add_value = value
                        add_i = i
        if add_i>=0: # checking if everything is fine and add this new condition
            if add_value not in MRS_new[add_i][add_attribute]:
                MRS_new[add_i][add_attribute].append(add_value)
            #if print_message:
            #    print('refine rules - add condition: add value {} to attribute {} in rule i = {}, score = {}'.format(add_value,add_attribute,add_i,score_max))

# Function to select the value to be added to the model when the random
# misclassified example is positive
# indices: rules that ex satisfies all but one condition
    def action_addvalue(self,MRS_new,ex_dict,Yhat,indices,q,print_message,df,Y):
        score_max = -1 # initialize/lower bound of score
        neg_ind = np.where(Yhat==0)[0] # search those predicted negative examples (FN that should be positives)
        add_val = -1 # initialize number of value to be added (last one) (?)
        #if print_message:
        #    print('\n add value ')
        for ind in indices: # per rule that ex satisfies all but one condition
        #######PARALLEL
            for att in MRS_new[ind].keys(): # per attribute in such rule
                if not any([x in ex_dict[att] for x in MRS_new[ind][att]]): 
                    # list: for the values of the att and the rule in the model,
                    # check if they are in the list of conditions that ex satisfies
                    # if they all are not, then save them as candidate values:
                    # si ninguno de los valores lo satisfacen de los que ya hay, incluirlos como candidatos
                    candidate_values = [x for x in ex_dict[att] if x not in MRS_new[ind][att]]
                    for val in candidate_values: # per candidate value
                        values = MRS_new[ind][att]+[val] # concatenate new values to older ones 
                        Z = (np.sum(df[:,values],axis = 1)>0).astype(int) # a vector of individuals saying if they satisfied the new values
                        if np.sum(np.asarray(Z))<self.N: # if the number of individuals that do satisfy it is less that the total...
                            #TP,FP,TN,FN = getConfusion(Z[neg_ind],Y[neg_ind]) # accuracy of such value and attr
                            #TN, FP, FN, TP = confusion_matrix(self.Y[neg_ind],Z[neg_ind]).ravel()
                            #######MODIFACTION
                            #score_temp = float(TP)/(TP+FP+1) # score from which we will select the best value
                            score_temp =np.dot(Z[neg_ind],Y[neg_ind])/(np.sum(Z[neg_ind])+1)
                            # update the best index (rule), attr and value to be added according to score
                            if score_temp>=score_max:
                                score_max = score_temp
                                add_ind = ind
                                add_att = att
                                add_val = val
        if score_max>0 and add_val in MRS_new[add_ind][add_att]: # if the value has already been, do nothing
            score_max = -1
        # if score_max continuous being -1 or add_val -1
        if add_val<0 or score_max == -1:
            ind_candidates = [i for i in range(len(MRS_new)) if i not in indices] # remainder rules that are not the ones in indices
            for ind in ind_candidates: # per rule
                for att in MRS_new[ind].keys(): # per attribute in rule
                    val_candidates = [x for x in self.attributeLevelIndeces[att] if x not in MRS_new[ind][att]] # values that are not the ones appearing already in such rule
                    for val in val_candidates: # per candidate value
                        values = MRS_new[ind][att]+[val] # concatenate new values to older ones 
                        Z = (np.sum(df[:,values],axis = 1)>0).astype(int) # a vector of individuals saying if they satisfied the new values
                        if np.sum(np.asarray(Z))<self.N: # if the number of individuals that do satisfy it is less that the total...
                            #TP,FP,TN,FN = getConfusion(Z[neg_ind],Y[neg_ind]) # accuracy of such value and attr
                            #TN, FP, FN, TP = confusion_matrix(self.Y[neg_ind],Z[neg_ind]).ravel()
                            #score_temp = float(TP)/(TP+FP+1) # score from which we will select the best value
                            score_temp =np.dot(Z[neg_ind],Y[neg_ind])/(np.sum(Z[neg_ind])+1)
                            # update the best index (rule), attr and value to be added according to score
                            if score_temp>=score_max:
                                score_max = score_temp
                                add_ind = ind
                                add_att = att
                                add_val = val
        if add_val>=0: # checking if everything is fine and add this new value
            MRS_new[add_ind][add_att].append(add_val)
            #if print_message:
            #    print('-- add value {} to feature \'{}\' in rule {} --'.format(add_val,add_att,add_ind))
        return                   

# Function to perturb randomly by deleting the worse rule in the rule set
#######
### W = W_curr (number of rules) x (number of individuals) indicator matrix
#######
    def action_perturb(self,MRS_new,W):
        #supp = np.array([sum(row) for row in W]) # support: number of examples satisfying rule set
        supp = np.sum(W, axis = 1)
        ind = np.where(supp == min(supp))[0][0] # index with rule with minimum support
        MRS_new.pop(ind) # delete such rule

# Function to try merge the new model
    def action_trymerge(self, MRS_new,df):
        np.random.seed(1)
        self.MRS_new = MRS_new # store MRS_new into the model
        l = len(MRS_new) # number of rules
        Z = np.array([[False]*len(self.attributeNames) for i in range(l)]) # initialize (number of rules in the rule set) x (number of attribute names) matrix
        # indicator matrix: 1 if rule i contains attribute j 
        for i,subdict in enumerate(MRS_new): # per rule in model
            Z[i][[int(att in MRS_new[i].keys()) for att in self.attributeNames]] = True
        dist = np.ones((l,l))*len(self.attributeNames) # (number of rules) x (number of rules) matrix which each element has the total number of attributes
        values = [] # initialize the list of values to (?)
        for i in range(l): # per rule
            for j in range(i+1,l,1): # per rule 
                # zip (a,b) is a list [(a1,b1),(a2,b2),...]
                # x^y is True if x or y is true but not both
                x = np.sum(np.asarray([x^y for x,y in zip(Z[i],Z[j])]))
                dist[i,j] = x # distance between rules
                values.append(x) # add distances values
        dist = dist/float(df.shape[1]) # normalize distance matrix with the number of total variables
        d = 2*(1+max(values) - min(values))/float(df.shape[1]) # WHAT IS THIS VALUE?
        dist = dist + np.multiply(d,np.random.rand(l,l))
        r1,r2 = np.unravel_index(dist.argmin(), dist.shape)
        attributes = [x for x in MRS_new[r1].keys() if (x in MRS_new[r2].keys() and set(MRS_new[r2][x])!=set(MRS_new[r1][x]))]
        if len(attributes)>0:
            att = sample(attributes,1)[0]
            vals = list(set(MRS_new[r2][att]).union(set(MRS_new[r1][att])))
            MRS_new[r2][att] = vals.copy()
            MRS_new[r1][att] = vals.copy()
            return
        if len(Z[r1])<=len(Z[r2]): # r1 is the shorter rule
            attributes = MRS_new[r1].keys()
            if len(attributes)>0:
                while True:
                    att = sample(attributes,1)[0]
                    attributes = [x for x in attributes if x!= att]
                    if att not in MRS_new[r2].keys():
                        MRS_new[r2][att] = MRS_new[r1][att]
                        break
                    else:
                        vals = list(set(MRS_new[r2][att]).union(set(MRS_new[r1][att])))
                        MRS_new[r2][att] = vals.copy()
                        MRS_new[r1][att] = vals.copy()
                        break
        else:
            attributes = MRS_new[r2].keys()
            if len(attributes)>0:
                while True:
                    att = sample(attributes,1)[0]
                    attributes = [x for x in attributes if x!= att]
                    if att not in MRS_new[r1].keys() :
                        MRS_new[r1][att] = MRS_new[r2][att]
                        break
                    else:
                        vals = list(set(MRS_new[r1][att]).union(set(MRS_new[r2][att])))
                        MRS_new[r2][att] = vals.copy()
                        MRS_new[r1][att] = vals.copy()
                        break

# Function to jump out of local minimum
#######
### W = W_curr
#######
    def action_jump(self,MRS_new,W,q,print_message,df,Y):
        seed(1)
        if random()<q: # randomly select a rule in the rule list to add a random condition
            #if print_message:
            #    print('in jump: Select a RANDOM item/condition to add to a RANDOM rule')
            add_i = sample(range(len(MRS_new)),1)[0] # random number of rule to be a condition added
            add_attribute = sample(set(self.attributeNames).difference(set(MRS_new[add_i].keys())),1)[0] # attribute to be added to rule selected but not repeated with any one
            add_value = sample(self.attributeLevelIndeces[add_attribute],1)[0] # random value to be added to attribute
        else:
            l = len(MRS_new)
            #if print_message:
            #    print('in jump: Select the BEST condition to add')
            score_max = -10000 # initialize/lower bound of score
            add_i = -1 # initialize number of rule to be a condition added (last rule)
            p_candidate = [] # initialization of list of candidate rules (?)
            for i in range(l): # per rule, save the number of FP
                TP,FP,TN,FN = getConfusion(W[i],Y)
                #TN, FP, FN, TP = confusion_matrix(self.Y,W[i]).ravel()
                p_candidate.append(FP)
            p_candidate = np.insert(p_candidate,0,0) # add 0 to list
            p_candidate = np.array(list(accumulate(p_candidate))) # accumulate
            if p_candidate[-1]==0: # if there is no FP in rules
                add_i = sample(range(l),1)[0] # random selection of index
            else: # if there is FP in rules
                add_i = find_lt(p_candidate,random()*p_candidate[-1])
            candidate_attributes = list(set(self.attributeNames).difference(set(MRS_new[add_i].keys())))
            # candidate attributes to be added to rule selected but not repeated with any one
            for attr in candidate_attributes: # per candidate attribute
                candidate_values = list(set(self.attributeLevelIndeces[attr])) # candidate values of each candidate attribute
                for value in candidate_values: # per value in candidate values
                    W_temp = W[add_i]*df[:,value] # vector of individuals satisfiying attr and values
                    #TP,FP,TN,FN = getConfusion(W_temp,Y) # accuracy of such value and attr
                    #TN, FP, FN, TP = confusion_matrix(self.Y,W_temp).ravel()
                    #score_temp = float(TP)/(TP+FP+1) # score from which we will select the best condition
                    score_temp =np.dot(W_temp,Y)/(np.sum(W_temp)+1)
                    # update the best condition (attribute + value) according to score
                    if score_temp>=score_max: 
                        score_max = score_temp
                        add_attribute = attr
                        add_value = value       
        if add_i>=0: # checking if everything is fine and add this new condition
            subdict = deepcopy(MRS_new[add_i])
            subdict[add_attribute].append(add_value)
            MRS_new.append(deepcopy(subdict))
        else:
            print('error!')
            
# Function to remove a condition that ex does not satify
    def action_rmcondition(self,MRS_new,V,W,ex_dict,print_message,df,Y):
        #if print_message:
        #    print('selected the BEST condition to remove')
        score_max = -10000000000000 # initialize/lower bound of score
        del_val = -1 # initialize number of value to be deleted (last one) (?)
        self.MRS_new = MRS_new # store MRS_new in the model
        self.V = V # store V in the model
        self.W = W # store W in the model
        self.ex_dict = ex_dict # store ex_dict in the model (conditions satisfied by ex)
        for i in range(len(MRS_new)): # per rule in the model
        ######PARA
            for att in MRS_new[i].keys(): # per attribute in the rule
                V_temp = deepcopy(V) # number of conditions satisfied in each rule
                W_temp = deepcopy(W) # indicator if a rule is satisfied
                if len(MRS_new[i][att])==1: #  if a condition has only one value
 #                   V_temp[i] = V[i] - self.df.ix[:,MRS_new[i][att][0]].values # compute the new number of conditions satisfied by the examples, assuming that we delete such condition
                    V_temp[i] = V[i] - df[:,MRS_new[i][att][0]] # compute the new number of conditions satisfied by the examples, assuming that we delete such condition
                    W_temp[i] = V_temp[i]==(len(MRS_new[i])-1) # compute the new indicator matrix where a rule is satisfied by the examples
                    Yhat_temp = (np.sum(W_temp,axis = 0)>0).astype(int) # predicted values without such condition
                    #TP,FP,TN,FN = getConfusion(Yhat_temp,Y) # accuracy that would be obtained
                    #TN, FP, FN, TP = confusion_matrix(self.Y,Yhat_temp).ravel()
                    #score_temp =  float(TP)/(TP+FP+1) # score according to whom the condition is going to be selected
                    score_temp =np.dot(Yhat_temp,Y)/(np.sum(Yhat_temp)+1)
                    self.score_temp = score_temp # store score_temp in the model
                    # update the best condition (attribute + value) according to score
                    if score_temp>=score_max:
                        score_max = score_temp
                        del_attr = att
                        del_i = i
                        del_val = -1 # only one value
                else: # if a condition has more that one value, remove a value
                    for val in MRS_new[i][att]: # per value in the attribute in the rule
                        vals = [v for v in MRS_new[i][att] if v != val] # list of the values except for the one we are interating on
                        V_temp[i] = V[i] - (np.sum(df[:,vals],axis = 1)>0).astype(int) # compute the new number of conditions satisfied by the examples, assuming that we delete such condition
                        W_temp[i] = V_temp[i]==(len(MRS_new[i])-1) # compute the new indicator matrix where a rule is satisfied by the examples
                        Yhat_temp = (np.sum(W_temp,axis = 0)>0).astype(int) # predicted values without such condition
                        #TP,FP,TN,FN = getConfusion(Yhat_temp,Y) # accuracy that would be obtained
                        #TN, FP, FN, TP = confusion_matrix(self.Y,Yhat_temp).ravel()
                        #score_temp =  float(TP)/(TP+FP+1) # score according to whom the condition is going to be selected
                        score_temp =np.dot(Yhat_temp,Y)/(np.sum(Yhat_temp)+1)
                        ######MODIFY
                        
                        # update the best condition (attribute + values) according to score
                        if score_temp>=score_max:
                            score_max = score_temp
                            del_attr = att
                            del_i = i
                            del_val = val
        if del_val == -1: # either remove a condition or remove a value
            MRS_new[del_i].pop(del_attr)
            if len(MRS_new[del_i])==0:
                MRS_new.pop(del_i)
        else: #(?)
            MRS_new[del_i][del_attr] = [x for x in MRS_new[del_i][del_attr] if x != del_val]
        #if print_message:
        #    print('action: remove attribute {} from rule {}'.format(del_attr,del_i) )    

# Function to remove a rule from the model when the misclassified example is negative
    ###NEW
    def action_rmrule(self,MRS_new,W,print_message,q,Y):
        if random()<q: # select randomly a rule within the pool
            del_i = sample(range(len(MRS_new)),1)[0]
        else: # select the best rule within the pool according to score
        ###
            all_sum = np.sum(W,axis = 0) # number of rules verified by each example
            score_max = -1 # initialize/lower bound of score
            del_i = -1 # initialize number of rule to be deleted (last one) (?)
            for i in range(len(MRS_new)): # per rule
                Yhat_temp= (all_sum - W[i]>0).astype(int) # new predicted values without such rule
                #TP,FP,TN,FN  = getConfusion(Yhat_temp,Y) # accuracy without such rule
                #TN, FP, FN, TP = confusion_matrix(self.Y,Yhat_temp).ravel()
                #score_temp = float(TP)/(TP+FP+1) # score according to whom the rule is going to be deleted
                score_temp =np.dot(Yhat_temp,Y)/(np.sum(Yhat_temp)+1)
                # update the best rule to delete according to score
                if score_temp>=score_max:
                    score_max = score_temp
                    del_i = i
        if del_i>=0:  # checking if everything is fine and remove rule
            MRS_new.pop(del_i)
        #if print_message:
        #    print('action4: remove rule {} from MRS'.format(del_i))

# Function to (randomly or not) select a rule to be inserted in the model that now cover ex, that is, a subset from ex_dict
    ###NEW
    def action_addrule(self,MRS_new,W,Yhat,ex,ex_dict,maxlen,q,print_message,iteration,Y,RMatrix):
    ###
        #if print_message:
        #    print('Add a BEST rule')
        score_max = -10000 # initialize/lower bound of score
        neg_ind = np.where(Yhat==0)[0] # search those predicted negative examples (FN that should be positives)
#        select = list(set(np.where(self.supp>self.minsupp[-1])[0]).intersection(np.where(self.RMatrix[ex,:]==1)[0]))
        # indices of rules to be selected to enter in game: conjunction of those generated and the ones satisfied by ex
        select = np.array(list(set(self.supp).intersection(np.where(self.RMatrix[ex,:]==1)[0]))) # RMatrix is the same of Z matrix just with the rules generated
        # select = [3,6,1,10]
        if len(select)>0:
            ###NEW
            #[[0,10],[10,30],[30,80]]
            maxlist = np.max(select) # 15
            maxlen_addrule = int(np.sum(maxlist >= np.array(self.lenindices)[:,0]))
            a = np.where(maxlist <= np.array(self.lenindices)[:,1])[0][0]
            whichlen = int(choices(range(a,maxlen_addrule),k=1)[0]) # len=2: [10,30]
            indices_select = select>=int(np.array(self.lenindices)[whichlen,0])
            select = select[indices_select]
            indices_select_2 = select<=int(np.array(self.lenindices)[whichlen,1])
            select = select[indices_select_2]
            select = list(select)
            ###
            ###NEW
            distributionrules = self.distributionrules[select]
            ###
            if random()<q: # select randomly a rule within the pool
                ###NEW
                if iteration > 100:
                   # print('iteration', iteration)
                    #print('distributionrules',np.sort(distributionrules)[-10:])
                    #print('length to consider',len(distributionrules))
                    add_rule_ind = choices(select,weights = distributionrules, k=1)[0]
                else:
                    add_rule_ind = sample(select,1)[0]
                ###
            else: # select the best rule within the pool according to score
                Yhat_neg_index = np.where(Yhat==0)[0] # indices of negative predicted values
                add_rule_ind = select[0] # initialize the rule to be selected
                for ind in select[1:]: # for all the indices except for the one fixed
                    #TP,FP,TN,FN = getConfusion(RMatrix[neg_ind,ind],Y[neg_ind]) # accuracy in those negative predicted values
                    #TN, FP, FN, TP = confusion_matrix(self.Y[neg_ind],self.RMatrix[neg_ind,ind]).ravel()
                    #score_temp = float(TP)/(TP+FP+1) # score according to whom the rule is going to be selected
                    #print("score temp",score_temp)
                    score_temp =np.dot(RMatrix[neg_ind,ind],Y[neg_ind])/(np.sum(RMatrix[neg_ind,ind])+1)
                    #print("score temp2",score_temp2)
                    # update the best rule to include according to score
                    if score_temp>=score_max:
                        score_max = score_temp
                        add_rule_ind = ind
                ###NEW
                self.distributionrules[add_rule_ind] = self.distributionrules[add_rule_ind] + 1
                ###
            new_dict = defaultdict(list) # initialize the new rule (dictionary) to be included
            rule = self.rules[add_rule_ind] # list of conditions of such rule
            for condition in rule: # per condition
                #try:
                Str_list=condition.split('_')
                att=Str_list[0]
                    #att,_,_ = condition.split('_')
                new_dict[att]= [self.colindices[condition]] # add this condition to the dictionary together with values
                # except:
                #     print(condition)
            MRS_new.append(new_dict) # add rule to dictionary

# Function to select the rule and the condition to be added to the model
# when the misclassified example is a negative one
    def action_addcondition(self,MRS_new,ex_dict,W,print_message,df,Y):
        #if print_message:
        #    print('Select the BEST condition to add')
        score_max = -1000000 # initialize/lower bound of score
        add_i = -1 # initialize number of value to be added (last one) (?)
        for i,subdict in enumerate(MRS_new): # per rule in the model
            # if (sum(W[i,np.where(self.Y==1)[0]])>self.minpossupp) and (sum(W[i,np.where(self.Y==0)[0]])<self.minnegsupp): # checking if the rule satisfies the minimum required in Theorems
            #if (np.sum(np.asarray(W[i,np.where(self.Y==1)[0]]))>self.minpossupp) and (np.sum(np.asarray(W[i,np.where(self.Y==0)[0]]))<self.minnegsupp):   
            candidate_attributes =[i for i in self.attributeNames if i not in subdict.keys()] # attributes that are not in the rule
            for attr in candidate_attributes: # per attribute in the candidate list
                candidate_values = list(set(self.attributeLevelIndeces[attr]).difference(set(ex_dict[attr]))) # values that belongs to the attribute but are not satisfied by the misclassified example
                for value in candidate_values: # per value in the candidate list
                    #TP,FP,TN,FN = getConfusion(W[i],Y) # accuracy current model
                    #TN, FP, FN, TP = confusion_matrix(self.Y,W[i]).ravel()
                    #start_time = time.time() 
                    #score1 = float(TP)/(TP+FP+1) # score current model
                    #score1time=time.time()- start_time
                    #print("score1 old", score1)
                    #print("score1 old time",score1time)
                    #start_time2 = time.time() 
                    
                    score1 =np.dot(W[i],Y)/(np.sum(W[i])+1)
                    # #scorenewtime=time.time()- start_time2
                    # print("score new",scorenew)
                    # print("score new time", scorenewtime)
                    ###### MODIFICATION  replace the prev row by dot(W,Y)/sum(W)
                    Z = W[i]*df[:,value] # W proposed model including this new value
                    TP,FP,TN,FN = getConfusion(Z,Y) # accuracy proposed model
                    #TN, FP, FN, TP = confusion_matrix(self.Y,Z).ravel()
                    #score2 = float(TP)/(TP+FP+1) # score proposed model
                    score2 =np.dot(Z,Y)/(np.sum(Z)+1)
                    ###### MODIFICATION  replace the prev row by dot(Z,Y)/sum(Z)


                    score_d = score2 - score1 # score difference between proposed and current models
                    # update the best value to include according to score
                    rule_minpossupp = self._minpossupp_rule(subdict, extra_len=1, extra_val=1)
                    if score_d>=score_max and (TP>rule_minpossupp and FP<self.minnegsupp):
                        score_max = score_d
                        add_attribute = attr
                        add_value = value
                        add_i = i
                        TP_max,FP_max,TN_max,FN_max = TP,FP,TN,FN
        if add_i>=0: # checking if everything is fine and add this new condition
            if add_value not in MRS_new[add_i][add_attribute]:
                MRS_new[add_i][add_attribute].append(add_value)
            #if print_message:
            #    print('action3 - add condition: add value {} to attribute {} in rule i = {}, score = {}, TP = {}, FP = {}, TN = {}, FN = {}'.format(add_value,add_attribute,add_i,score_max,TP_max,FP_max,TN_max,FN_max))

# Function to compute the information about the new solution
# MRS_new: rule set (list of dictionaries)
    def compute_prob(self, MRS_new,df,Y):
        #######
        ### MRS = MRS_new
        #######
        ###  subdict = defaultdict(list)
        ###  subdict['Amountdefenderfeesassessed'].append(482)
        ###  subdict['Recodeofcompliancepercent'].append(162)
        ###  MRS_new.append(subdict)
        #######
        ### I think that V and W could be computed faster
        #######
        MRS_merged= self.merge(MRS_new) # merge MRS_new
        MRS_new_rlens = [len(subdict) for subdict in MRS_merged] # number of conditions per rule
        MRS_new_len = len(MRS_new_rlens) # number of rules
        MRS_new_nterms = self.calculate_items(MRS_merged)
        W = np.zeros((MRS_new_len,len(Y))) # matrix (number of rules)x(examples): 1 if an example satisfies a rule
        V = np.zeros((MRS_new_len,len(Y))) # matrix (number of rules)x(examples): int with how many conditions each example satisfies in a rule 
        rm_attributes_list = [] # list of attributes to remove from model
        for i,subdict in enumerate(MRS_merged): # per rule in rule set
            for att in subdict.keys(): # per attribute in rule
                tmp = (np.sum(df[:,subdict[att]],axis = 1)>0).astype(int)
                # N vector saying if att is satisfied per individual (1) or not (0)
                V[i] += tmp # add 1 for each att satisfied
                if np.mean(tmp)==1: # if all examples verify 1 rule, remove attribute 
                    rm_attributes_list.append((i,att))
            W[i] = (V[i]==MRS_new_rlens[i]).astype(int) # 1 if the number of conditions per rule satisfied by a examples is the total length of such rule
        for i,att in rm_attributes_list: # remove those attributes that are verified by all samples
            MRS_merged[i].pop(att,None)
        Yhat = (np.sum(W,axis = 0)>0).astype(int) # predict 1 if at least one rule in the rule list is satisfied by individual
        #######
        ### Y = self.Y
        #######
        TP,FP,TN,FN = getConfusion(Yhat,Y) # number (not percetage) of TP, FP, TN and FN
        #TN, FP, FN, TP = confusion_matrix(self.Y,Yhat).ravel()
        Error = (FP + FN)/len(Y) # first term objective (accuracy)
        NumOfLiterals = np.sum(MRS_new_rlens) # second term objective (interpretability)
        NumOfRules = MRS_new_len # third term objective (interpretability)
        NumOfTerms = MRS_new_nterms
        pt_new = Error + self.c1*NumOfLiterals + self.c2*NumOfRules + self.c3*NumOfTerms # objective value
        return MRS_merged,[TP,FP,TN,FN],[Error,NumOfLiterals,NumOfRules,NumOfTerms],Yhat,W,V,MRS_new_rlens,MRS_new_len

# Function to merge rules if needed (?) redundant information, let's say
# MRS: rule set (list of dictionaries)
# Some notes:
# 1. If RuleA c RuleB -> delete RuleB
# 2. If nothing repeated, order rules by length
# 3. If I add a rule with one single condition repeated, delete the other rule and
# maintain this one
# 4. If there are 2 rules with single conditions (same) but different values, merge them
    def merge(self,MRS):
        MRS_merged = [deepcopy(subdict) for subdict in MRS] # copy of MRS
        MRS_merged = sorted(MRS_merged,key = lambda x:len(x)) # order MRS from lower to higher length
        self.MRS_merged = MRS_merged # add MRS_merged to model
        while True:
            rm_list = [] # initialization of list of indices to remove (?)
            ########
            ### row_indexA = 0
            ########
            for row_indexA in range(len(MRS_merged)): # for each rule
                self.row_indexA = row_indexA # add index to model
                if row_indexA >= len(MRS_merged): # AFTER TONG MEETING
                    break
                attributeA = set(MRS_merged[row_indexA].keys()) # set of attributes in each rule
                if row_indexA not in rm_list: # if index row_index is not in rm_list
                    #######
                    ### row_indexB = 1
                    #######
                    for row_indexB in range(len(MRS_merged)): # for another rule in the rule list different from the above
                        if row_indexB != row_indexA and row_indexB not in rm_list: # if another index is not also in rm_list
                            attributeB = set(MRS_merged[row_indexB].keys()) # set of attributes in each rule
                            if attributeA.issubset(attributeB): # if set of attributes A is a subset of the set of attributes B (recall that they where ordered according to length)
                                subset = 0 # initialization of (?)
                                same = 0 # initialization of (?)
                                ########
                                ### attr = list(attributeA)[0]
                                ########
                                for attr in attributeA: # for each attribute in the set of attributes A
                                    if set(MRS_merged[row_indexB][attr])<=set(MRS_merged[row_indexA][attr]):
                                        # if the set of values of attributeB is a subset of the set of values of attribute, 
                                        # add 1 to subset
                                        subset += 1
                                        if set(MRS_merged[row_indexB][attr])==set(MRS_merged[row_indexA][attr]):
                                        # if the set of values of attributeB is equal to the set of values of attributeA,
                                        # add 1 to same
                                            same += 1
                                if same==(len(attributeA)-1) and len(attributeA)==len(attributeB): # merge values into attributeA and delete attributeB
                                    # print('merging index_A = {}, index_B = {}'.format(row_indexA, row_indexB))                
                                    for attr in attributeA:
                                        MRS_merged[row_indexA][attr] = list(set(MRS_merged[row_indexA][attr]).union(set(MRS_merged[row_indexB][attr])))
                                    rm_list.append(row_indexB)           
                                elif subset == len(attributeA):  # delete attributeB directly because they are the same
                                    # print('deleting index_A = {}, index_B = {}'.format(row_indexA, row_indexB)) 
                                    rm_list.append(row_indexB) 
                        row_indexB += 1
                    rm = []
                #     print('A = {}, B = {}, len = {}'.format(row_indexA,row_indexB,len(MRS_merged)))
                    for attr in MRS_merged[row_indexA].keys(): # per attribute per rule in MRS_merged
                        values = MRS_merged[row_indexA][attr] # values of such attribute
                        self.values = values # save values into model
                        self.MRS_merged = MRS_merged
                        if sum(np.sum(self.df[:,values],axis = 1)>0) == self.N:
                        # if len(MRS_merged[row_indexA][attr])==self.attributeLevelNum[attr]:
                            rm.append(attr)
                    for attr in rm:
                        MRS_merged[row_indexA].pop(attr, None) # delete attr 
                            # del MRS_merged[row_indexA][attr]
                    if len(MRS_merged[row_indexA])==0:
                        MRS_merged.pop(row_indexA)
                    row_indexA += 1
            MRS_merged = [subdict for ind,subdict in enumerate(MRS_merged) if ind not in rm_list]
            if len(rm_list)==0:
                break
        return MRS_merged

# Function to print the model
    def printMRS(self,MRS):
        for i,subdict in enumerate(MRS): # per rule in rule set and index
            rule = 'rule {}:'.format(i) # print number of rule
            for key in subdict.keys(): # per attr/condition in rule
                rule += '('+str(key)+':' # print attr
                values_ls=[]
                for value in subdict[key]:
                    val=self.itemNames[value].split('_')
                    val.pop(0)
                    Cond='_'.join(val)
                    values_ls.append(Cond)
                #values = [self.itemNames[value].split('_')[1:] for value in subdict[key]] # save value of attr
                values = ' or '.join(values_ls) # disjunction of values or a single one
                rule = rule + values + '),' # join everything
            print(rule) # print rule

# Function to compute the number of items in the rule set
    def calculate_items(self,MRS):
        count_items=0
        for i,subdict in enumerate(MRS): # per rule in rule set and index
            rule = 'rule {}:'.format(i) # print number of rule
            for key in subdict.keys(): # per attr/condition in rule
                rule += '('+str(key)+':' # print attr
                values = [self.itemNames[value].split('_')[1] for value in subdict[key]]
 # save value of attr
                count_items=count_items+len(values)
                values = ' or '.join(values) # disjunction of values or a single one
                rule = rule + values + '),' # join everything
        return count_items