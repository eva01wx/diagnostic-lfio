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

# Definition of the class object MRS
class ORS(object):   
    
# Function to initialize MRS
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
        self.Rset= []
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
        # Theorem 2 (conservative static bound, assumes len=val=1):
        self.minpossupp = np.ceil((self.c1 + self.c2 + self.c3)*self.N)

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
            pindex = np.where(self.Y==1)[0] # indices of positive examples
            start_time = time.time()
            rules=[]
            start_time = time.time()
            self.nrulesperlength = [] # [10,20,50] Total = 80
            self.lenindices = [] # [[0,10],[10,30],[30,80]]
            ii = 0
            for z in range(1,maxlen+1):
                # Theorem 2 with len(r)=val(r)=z: FP-growth produces binary
                # single-value items, so each condition contributes exactly one value.
                minpossupp = self._minpossupp(z, z)
                rules_ls = fpgrowth([itemMatrix[i] for i in pindex],supp = np.max(np.asarray([5,100*minpossupp/len(pindex)])),zmin = z,zmax = z)
                #print("z size",z)
                rules=rules+rules_ls
                hmr = len(rules_ls)
                self.nrulesperlength.append(hmr)
                self.lenindices.append([ii,ii + hmr])
                ii = ii + hmr
            self.distributionrules = np.zeros(len(rules)) +  0.001
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
            for length in [maxlen]:
                n_estimators = 500
                clf = RandomForestClassifier(n_estimators = n_estimators,max_depth = length, random_state=1)
                clf.fit(self.df,self.Y)
                for n in range(n_estimators):
                    rules.extend(extract_rules(clf.estimators_[n],self.col_ls))
                    # extracting rules for each tree and each RF
            rules = [list(x) for x in set(tuple(np.sort(x)) for x in rules)] #sort rules
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
        #print("Print time main--------------------------------------------")
        #print('\tTook %0.3fs to generate %d rules' % (self.generate_time, len(rules)))
        #print("-----------------------------------------------------------")
        index = []
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
        n_cores=multiprocessing.cpu_count()
        self.N_part=math.floor(N_rules/n_cores)
        shuffle(rules)
        rules=np.array(rules,dtype=object)
        rl=np.array_split(rules, n_cores)
        start_time=time.time()
        rule_ls, RMatrix_ls,supp_ls = zip(*Parallel(n_jobs=-1)(delayed(self.screen_rules)(r) for r in rl))
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
        indptr = list(accumulate(len_rules)) # accumulated sum of the lengths of rules generated
        indptr.insert(0,0) # adding 0 as the first argument
        indptr = np.array(indptr) # indices for columns in ruleMatrix
        data = np.ones(len(indices)) # array of 1s of length the number of indices
        ruleMatrix = csc_matrix((data,indices,indptr),shape = (len(self.col_ls),len(rules)))
        mat = np.matrix(self.df) * ruleMatrix 
        lenMatrix = np.matrix([len_rules for i in range(self.df.shape[0])])
        Z =  (mat ==lenMatrix).astype(int) 
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
        supp_select = np.arange(0,len(TP))
        if len(supp_select)<=self.N_part: # if the number of rules generated is less than N, DONE, select all the rules generated.
            rules = [rules[i] for i in supp_select] # list of rules (self.rules = rules)
            RMatrix = np.array(Z[:,supp_select]) # matrix Z (self.RMatrix = Z)
            rules_len = [len(set([name.split('_')[0] for name in rule])) for rule in rules] # vector len_rules (self rules_len = len_rules)
            supp = np.array(np.sum(Z,axis=0).tolist()[0])[supp_select] 
        else: # if not, select N according to criteria
            FP = np.array(np.sum(Z,axis = 0))[0] - TP # FP: P - TP, per rule generated
            TN = len(self.Y) - np.sum(self.Y) - FP # TN: (N - pos) - FP, per rule generated
            FN = np.sum(self.Y) - TP # FN: positive uncorrectly classified, per rule generated
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
       
        return rules, RMatrix,supp

# Function to train MRS using SA
# Niteration: number of iterations in each chain in SA
# q: probability of being random
# print_message: show some messages (True) or not (False)
# init: initial rule set
    def train(self,Niteration = 2000, q=0.3, print_message=True,init = []):
        #bound_count=0
        #RS=RS+1
        #Rset_Remaining=RS
        #self.precompute()
        start_time = time.time()
        self.Y = np.array(self.Y)
        T0=20
        pt_max = 100000000 # initial objective
        MRS_max = []
        self.maps=[] # list of ?
        self.objvalues = []
        if len(self.rules) == 0:
            # No rule satisfies the Theorem-1 minimum-negative-support bound:
            # c1+c2+c3 is so large that the empty rule set (which predicts
            # class 0) is optimal. The Rashomon set is also empty by the same
            # argument -- no non-empty rule set can match its objective.
            import warnings
            warnings.warn(
                "Regularization (c1+c2+c3) is too large: no rule satisfies the "
                "minimum negative-support bound (minnegsupp = "
                + str(self.minnegsupp) + "). Returning an empty rule set "
                "(predicts class 0 for every example) and an empty Rashomon set.",
                stacklevel=2,
            )
            self.Rset = []
            return [], self.Rset, self.maps
        if init == []: # if no rules were generated?
            seed(1)
            np.random.seed(1)
            N = sample(range(1,8,1),1)[0] # random number between 1 and 7. I guess random number of rules.
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
        else:
            MRS_curr = init # if initial rule list is directly given
        self.MRS_curr = MRS_curr # save MRS_curr in model
        MRS_curr,_,prob,Yhat_curr,W_curr,V_curr,MRS_curr_rlens,MRS_curr_len = self.compute_prob(MRS_curr) # compute current solution accuracy
        pt_curr = prob[0] + self.c1*prob[1] + self.c2*prob[2] + self.c3*prob[3] # current objective value
        self.dynamicmaxrulesetsize = [pt_curr/(self.c1+self.c2+self.c3)]
        self.objvalues = []
        ntimes_mis = np.zeros(self.N)
        for iter in range(Niteration): 
            if iter>0.9*Niteration:
                Yhat_curr,pt_curr,W_curr,V_curr,MRS_curr_rlens,MRS_curr_len = Yhat_max.copy(),pt_max,W_max.copy(),V_max.copy(),MRS_max_rlens.copy(),MRS_max_len
                MRS_curr = [subdict for subdict in MRS_max]
            MRS_new = [deepcopy(subdict) for subdict in MRS_curr] # save MRS_curr in MRS_new
            if print_message:
                self.printMRS(MRS_new)
            added_rule = False # initialize value to False if no rule has been added to rule list
            _,_,temp,_,_,_,_,_ = self.compute_prob(MRS_new)
            self.objvalues.append(temp[0] + self.c1*temp[1] + self.c2*temp[2] + self.c3*temp[3])

            logical_incorr = self.Y!=Yhat_curr
            incorr = np.where(logical_incorr)[0]
            if len(incorr)==0: # every data point is correctly classified
                trim = False # initialize trim to False (?)
                break # finish
                # it means the MRS correctly classified all points but the rule could be redundant, so triming is needed
            else: # some example is misclassified
                trim = False
                ntimes_mis = np.add(ntimes_mis,logical_incorr.astype(int))
                if iter > 30:
                    misc=ntimes_mis[logical_incorr]/(iter+1)
                    wei=0.5-np.abs(0.5-misc)+1e-7
                    c = choices(range(len(incorr)), weights=wei, k=1)[0]
                    ex=incorr[c]
                else:
                    ex = choices(list(incorr), k=1)[0]
                ex_dict = defaultdict(list) # dictionary with conditions that ex satisfies
                for i,name in enumerate(self.col_ls):
                    attr_name = name.split('_')[0]
                    if self.df[ex,i]==1:
                        ex_dict[attr_name].append(i)
            if random()<0 and MRS_curr_len>1:
                self.action_perturb(MRS_new,W_curr)
            else:
                if trim == False and random()<=0: # jump out of local minimum:
                    self.action_jump(MRS_new,W_curr,q,print_message)
                elif trim==False and self.Y[ex]==1: # if the misclassified example is positive 
                    # it means the model failed to cover it, so we need to do one of three following
                    t = random()
                    # Deactivate the option of adding a rule if the maximum rule set size is overcome
                    if MRS_curr_len >= self.dynamicmaxrulesetsize[-1]:
                        #bound_count=bound_count+1
                        t = (2/3)*t
                    # 1) add value to cover ex with probability 1/3
                    if t<(1/3): 
                        # find out which of the rules does ex satisfy all but one condition
                        temp  = [i for i in range(MRS_curr_len) if V_curr[i][ex] == MRS_curr_rlens[i]-1]
                        self.action_addvalue(MRS_new,ex_dict,Yhat_curr,temp,q,print_message)
                    # 2) remove condition (from a rule) that ex does not satisfy with probability 1/3
                    elif t<(2/3) and not (MRS_curr_len==1 and len(MRS_new[0])==1) and MRS_curr_len>0: # do in case that we have at least 1 rule with 2 conditions
                        self.action_rmcondition(MRS_new,V_curr,W_curr,ex_dict,print_message) # problematic to do that, DO NOT REMOVE CONDITION
                # 3) add a rule that ex satisfies
                    else: 
                        self.action_addrule(MRS_new,W_curr,Yhat_curr,ex,ex_dict,self.maxlen,q,print_message,iter)
                        added_rule = True
                else: # the misclassified example is negative
                    # it means the model covers the wrong data, so we need to do one of the following 
                    t = random()
                    # 1) Add a condition to a rule that satisfies the example that DOES NOT satisfies such condition
                    if trim==False and (MRS_curr_len<=1 or t<=1/3):#or (random()<=1.0/(2-float(iter)/Niteration) and MRS_curr_len<=10: 
                        self.action_addcondition(MRS_new,ex_dict,W_curr,print_message)
                    # 2) Remove a rule satisfied by the misclassified example
                    elif t<= (2/3): 
                        self.action_rmrule(MRS_new,W_curr,print_message,q)
                    # 3) Replace a rule: remove + add
                    else: 
                        self.action_rmrule(MRS_new,W_curr,print_message,q) # remove a rule
                        MRS_new,cfmatrix,prob,Yhat_new,W_new,V_new,MRS_new_rlens,MRS_new_len = self.compute_prob(MRS_new) # compute current solution accuracy
                        incorr = np.where(np.logical_and(self.Y==1, Yhat_new==0))[0] # indices of misclassified examples
                        if len(incorr)>0:
                            ex = sample(list(incorr),1)[0] # a random misclassified example
                            ex_dict = defaultdict(list) # dictionary with conditions that ex satisfies
                            for i,name in enumerate(self.col_ls):
                                attr_name = name.split('_')[0]
                                if self.df[ex,i]==1:
                                    ex_dict[attr_name].append(i)
                            self.action_addrule(MRS_new,W_new,Yhat_new,ex,ex_dict,self.maxlen,q,print_message,iter) # add a rule
                            added_rule = True # save if a rule has been added to the model
            MRS_new,cfmatrix,prob,Yhat_new,W_new,V_new,MRS_new_rlens,MRS_new_len= self.compute_prob(MRS_new) # compute current solution accuracy
            if random()<0.25 and added_rule == True:
                self.refine_rule(MRS_new,W_new,print_message) # add a condition to a rule
                MRS_new,cfmatrix,prob,Yhat_new,W_new,V_new,MRS_new_rlens,MRS_new_len= self.compute_prob(MRS_new) # compute current solution accuracy
            
            
            pt_new = prob[0] + self.c1*prob[1] + self.c2*prob[2] + self.c3*prob[3] # new objective value
            T = np.true_divide(T0, 1 + iter) # decreasing temperature
            TP_new,FP_new,TN_new,FN_new = cfmatrix
            Error_new = (FP_new + FN_new)/len(self.Y)
            #if  iter > Niteration-Rset_Remaining-1:
            self.Rset.append([MRS_new,pt_new,1-Error_new])
            #Rset_Remaining=RS-len(self.Rset)+1
        
            alpha = np.exp(float(pt_curr -pt_new)/T) # probability of accepting new solution though worse
            if pt_new < pt_max: # if the new solution generated is better that the current one, update it by the new one in MRS_max
                TP,FP,TN,FN = cfmatrix
                tpr = float(TP)/(TP+FN)
                fpr = float(FP)/(FP+TN)
                MRS_max = [deepcopy(subdict) for subdict in MRS_new]
                pt_max,Yhat_max, W_max,V_max, MRS_max_rlens, MRS_max_len = pt_new, Yhat_new.copy(),W_new.copy(),V_new.copy(),MRS_new_rlens.copy(),MRS_new_len # save parameters of model MRS_max
                self.MRS_max = MRS_max # save MRS_max into the model
                #if iter>RP*Niteration:
                #    Error = (FP + FN)/len(self.Y)
                #    self.Rset.append([MRS_max,pt_max,1-Error])
                #    Rset_Remaining=RS-len(self.Rset)+1
                #    if len(self.Rset)>RS:
                #        _=self.Rset.pop(0)
                self.dynamicmaxrulesetsize.append(pt_new/(self.c1+self.c2+self.c3)) #Theorem4
            number = random()
            if number <= alpha: # if a random number is less or equal than alpha, we accept the new model though it is not that good
                MRS_curr = [deepcopy(subdict) for subdict in MRS_new]
                Yhat_curr,pt_curr,W_curr,V_curr,MRS_curr_rlens,MRS_curr_len = Yhat_new.copy(),pt_new,W_new.copy(),V_new.copy(),MRS_new_rlens.copy(),MRS_new_len

        MRS_merged = self.merge(MRS_max) # merge model MRS_max
        return MRS_merged, self.Rset, self.maps #bound_count # return the new model together with some information stored in maps                 

# Function to add a condition to a rule in the model
    def refine_rule(self,MRS_new,W,print_message):
        score_max = 0 # initialize/lower bound of score
        add_i = -1 # initialize number of value to be added (last one) (?)
        for i,subdict in enumerate(MRS_new): # per rule in the model
            # It is always verified because of the way we premine rules
            candidate_attributes =[i for i in self.attributeNames if i not in subdict.keys()] # attributes that are not in the rule
            for attr in candidate_attributes: # per attribute in the candidate list
                candidate_values = self.attributeLevelIndeces[attr] # values that belongs to the attribute
                for value in candidate_values: # per value in the candidate list
                    score1 =np.dot(W[i],self.Y)/(np.sum(W[i])+1)
                    Z = W[i]*self.df[:,value] # W proposed model including this new value
                    TP,FP,TN,FN = getConfusion(Z,self.Y) # accuracy proposed model
                    score2 =np.dot(Z,self.Y)/(np.sum(Z)+1)
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
           
# Function to select the value to be added to the model when the random
# misclassified example is positive
# indices: rules that ex satisfies all but one condition
    def action_addvalue(self,MRS_new,ex_dict,Yhat,indices,q,print_message):
        score_max = -1 # initialize/lower bound of score
        neg_ind = np.where(Yhat==0)[0] # search those predicted negative examples (FN that should be positives)
        add_val = -1 # initialize number of value to be added (last one) (?)
        for ind in indices: # per rule that ex satisfies all but one condition
            for att in MRS_new[ind].keys(): # per attribute in such rule
                if not any([x in ex_dict[att] for x in MRS_new[ind][att]]): 
                    # list: for the values of the att and the rule in the model,
                    # check if they are in the list of conditions that ex satisfies
                    # if they all are not, then save them as candidate values:
                    # si ninguno de los valores lo satisfacen de los que ya hay, incluirlos como candidatos
                    candidate_values = [x for x in ex_dict[att] if x not in MRS_new[ind][att]]
                    for val in candidate_values: # per candidate value
                        values = MRS_new[ind][att]+[val] # concatenate new values to older ones 
                        Z = (np.sum(self.df[:,values],axis = 1)>0).astype(int) # a vector of individuals saying if they satisfied the new values
                        if np.sum(np.asarray(Z))<self.N: # if the number of individuals that do satisfy it is less that the total...
                            score_temp =np.dot(Z[neg_ind],self.Y[neg_ind])/(np.sum(Z[neg_ind])+1)
                            # update the best index (rule), attr and value to be added according to score
                            if score_temp>=score_max:
                                score_max = score_temp
                                add_ind = ind
                                add_att = att
                                add_val = val
        if score_max>0 and add_val in MRS_new[add_ind][add_att]: # if the value has already been, do nothing
            score_max = -1
        if add_val<0 or score_max == -1:
            ind_candidates = [i for i in range(len(MRS_new)) if i not in indices] # remainder rules that are not the ones in indices
            for ind in ind_candidates: # per rule
                for att in MRS_new[ind].keys(): # per attribute in rule
                    val_candidates = [x for x in self.attributeLevelIndeces[att] if x not in MRS_new[ind][att]] # values that are not the ones appearing already in such rule
                    for val in val_candidates: # per candidate value
                        values = MRS_new[ind][att]+[val] # concatenate new values to older ones 
                        Z = (np.sum(self.df[:,values],axis = 1)>0).astype(int) # a vector of individuals saying if they satisfied the new values
                        if np.sum(np.asarray(Z))<self.N: # if the number of individuals that do satisfy it is less that the total...
                            score_temp =np.dot(Z[neg_ind],self.Y[neg_ind])/(np.sum(Z[neg_ind])+1)
                            # update the best index (rule), attr and value to be added according to score
                            if score_temp>=score_max:
                                score_max = score_temp
                                add_ind = ind
                                add_att = att
                                add_val = val
        if add_val>=0: # checking if everything is fine and add this new value
            MRS_new[add_ind][add_att].append(add_val)
        return                   

# Function to perturb randomly by deleting the worse rule in the rule set
    def action_perturb(self,MRS_new,W):
        supp = np.sum(W, axis = 1)
        ind = np.where(supp == min(supp))[0][0] # index with rule with minimum support
        MRS_new.pop(ind) # delete such rule

# Function to try merge the new model
    def action_trymerge(self, MRS_new):
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
                x = np.sum(np.asarray([x^y for x,y in zip(Z[i],Z[j])]))
                dist[i,j] = x # distance between rules
                values.append(x) # add distances values
        dist = dist/float(self.df.shape[1]) # normalize distance matrix with the number of total variables
        d = 2*(1+max(values) - min(values))/float(self.df.shape[1]) # WHAT IS THIS VALUE?
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
    def action_jump(self,MRS_new,W,q,print_message):
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
                TP,FP,TN,FN = getConfusion(W[i],self.Y)
                p_candidate.append(FP)
            p_candidate = np.insert(p_candidate,0,0) # add 0 to list
            p_candidate = np.array(list(accumulate(p_candidate))) # accumulate
            if p_candidate[-1]==0: # if there is no FP in rules
                add_i = sample(range(l),1)[0] # random selection of index
            else: # if there is FP in rules
                add_i = find_lt(p_candidate,random()*p_candidate[-1]) # WHY RANDOM
            candidate_attributes = list(set(self.attributeNames).difference(set(MRS_new[add_i].keys())))
            # candidate attributes to be added to rule selected but not repeated with any one
            for attr in candidate_attributes: # per candidate attribute
                candidate_values = list(set(self.attributeLevelIndeces[attr])) # candidate values of each candidate attribute
                for value in candidate_values: # per value in candidate values
                    W_temp = W[add_i]*self.df[:,value] # vector of individuals satisfiying attr and values
                    score_temp =np.dot(W_temp,self.Y)/(np.sum(W_temp)+1)
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
    def action_rmcondition(self,MRS_new,V,W,ex_dict,print_message):
        #if print_message:
        #    print('selected the BEST condition to remove')
        score_max = -10000000000000 # initialize/lower bound of score
        del_val = -1 # initialize number of value to be deleted (last one) (?)
        self.MRS_new = MRS_new # store MRS_new in the model
        self.V = V # store V in the model
        self.W = W # store W in the model
        self.ex_dict = ex_dict # store ex_dict in the model (conditions satisfied by ex)
        for i in range(len(MRS_new)): # per rule in the model
            for att in MRS_new[i].keys(): # per attribute in the rule
                V_temp = deepcopy(V) # number of conditions satisfied in each rule
                W_temp = deepcopy(W) # indicator if a rule is satisfied
                if len(MRS_new[i][att])==1: #  if a condition has only one value
 #                   V_temp[i] = V[i] - self.df.ix[:,MRS_new[i][att][0]].values # compute the new number of conditions satisfied by the examples, assuming that we delete such condition
                    V_temp[i] = V[i] - self.df[:,MRS_new[i][att][0]] # compute the new number of conditions satisfied by the examples, assuming that we delete such condition
                    W_temp[i] = V_temp[i]==(len(MRS_new[i])-1) # compute the new indicator matrix where a rule is satisfied by the examples
                    Yhat_temp = (np.sum(W_temp,axis = 0)>0).astype(int) # predicted values without such condition
                    score_temp =np.dot(Yhat_temp,self.Y)/(np.sum(Yhat_temp)+1) # score according to whom the condition is going to be selected
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
                        V_temp[i] = V[i] - (np.sum(self.df[:,vals],axis = 1)>0).astype(int) # compute the new number of conditions satisfied by the examples, assuming that we delete such condition
                        W_temp[i] = V_temp[i]==(len(MRS_new[i])-1) # compute the new indicator matrix where a rule is satisfied by the examples
                        Yhat_temp = (np.sum(W_temp,axis = 0)>0).astype(int) # predicted values without such condition
                        score_temp =np.dot(Yhat_temp,self.Y)/(np.sum(Yhat_temp)+1)
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
        else: 
            MRS_new[del_i][del_attr] = [x for x in MRS_new[del_i][del_attr] if x != del_val]

# Function to remove a rule from the model when the misclassified example is negative
    def action_rmrule(self,MRS_new,W,print_message,q):
        if random()<q: # select randomly a rule within the pool
            del_i = sample(range(len(MRS_new)),1)[0]
        else: # select the best rule within the pool according to score
            all_sum = np.sum(W,axis = 0) # number of rules verified by each example
            score_max = -1 # initialize/lower bound of score
            del_i = -1 # initialize number of rule to be deleted (last one) (?)
            for i in range(len(MRS_new)): # per rule
                Yhat_temp= (all_sum - W[i]>0).astype(int) # new predicted values without such rule
                score_temp =np.dot(Yhat_temp,self.Y)/(np.sum(Yhat_temp)+1) # score according to whom the rule is going to be deleted
                # update the best rule to delete according to score
                if score_temp>=score_max:
                    score_max = score_temp
                    del_i = i
        if del_i>=0:  # checking if everything is fine and remove rule
            MRS_new.pop(del_i)
        #if print_message:
        #    print('action4: remove rule {} from MRS'.format(del_i))

# Function to (randomly or not) select a rule to be inserted in the model that now cover ex, that is, a subset from ex_dict
    def action_addrule(self,MRS_new,W,Yhat,ex,ex_dict,maxlen,q,print_message,iteration):
        score_max = -10000 # initialize/lower bound of score
        neg_ind = np.where(Yhat==0)[0] # search those predicted negative examples (FN that should be positives)
        select = np.array(list(set(self.supp).intersection(np.where(self.RMatrix[ex,:]==1)[0]))) # RMatrix is the same of Z matrix just with the rules generated
        if len(select)>0:
            maxlist = np.max(select) # 15
            maxlen_addrule = int(np.sum(maxlist >= np.array(self.lenindices)[:,0]))
            a = np.where(maxlist <= np.array(self.lenindices)[:,1])[0][0]
            whichlen = int(choices(range(a,maxlen_addrule),k=1)[0])
            indices_select = select>=int(np.array(self.lenindices)[whichlen,0])
            select = select[indices_select]
            indices_select_2 = select<=int(np.array(self.lenindices)[whichlen,1])
            select = select[indices_select_2]
            select = list(select)
            distributionrules = self.distributionrules[select]
            if random()<q: # select randomly a rule within the pool
                if iteration > 100:
                    add_rule_ind = choices(select,weights = distributionrules, k=1)[0]
                else:
                    add_rule_ind = sample(select,1)[0]
                ###
            else: # select the best rule within the pool according to score
                Yhat_neg_index = np.where(Yhat==0)[0] # indices of negative predicted values
                add_rule_ind = select[0] # initialize the rule to be selected
                for ind in select[1:]: # for all the indices except for the one fixed
                    score_temp =np.dot(self.RMatrix[neg_ind,ind],self.Y[neg_ind])/(np.sum(self.RMatrix[neg_ind,ind])+1)
                    # update the best rule to include according to score
                    if score_temp>=score_max:
                        score_max = score_temp
                        add_rule_ind = ind
                self.distributionrules[add_rule_ind] = self.distributionrules[add_rule_ind] + 1
            new_dict = defaultdict(list) # initialize the new rule (dictionary) to be included
            rule = self.rules[add_rule_ind] # list of conditions of such rule
            for condition in rule: # per condition
                Str_list=condition.split('_')
                att=Str_list[0]
                new_dict[att]= [self.colindices[condition]] # add this condition to the dictionary together with values
            MRS_new.append(new_dict) # add rule to dictionary

# Function to select the rule and the condition to be added to the model
# when the misclassified example is a negative one
    def action_addcondition(self,MRS_new,ex_dict,W,print_message):
        #if print_message:
        #    print('Select the BEST condition to add')
        score_max = -1000000 # initialize/lower bound of score
        add_i = -1 # initialize number of value to be added (last one) (?)
        for i,subdict in enumerate(MRS_new): # per rule in the model
            candidate_attributes =[i for i in self.attributeNames if i not in subdict.keys()] # attributes that are not in the rule
            for attr in candidate_attributes: # per attribute in the candidate list
                candidate_values = list(set(self.attributeLevelIndeces[attr]).difference(set(ex_dict[attr]))) # values that belongs to the attribute but are not satisfied by the misclassified example
                for value in candidate_values: # per value in the candidate list
                    score1 =np.dot(W[i],self.Y)/(np.sum(W[i])+1)
                    Z = W[i]*self.df[:,value] # W proposed model including this new value
                    TP,FP,TN,FN = getConfusion(Z,self.Y) # accuracy proposed model
                    score2 =np.dot(Z,self.Y)/(np.sum(Z)+1)
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
            
# Function to compute the information about the new solution
# MRS_new: rule set (list of dictionaries)
    def compute_prob(self, MRS_new):
        ### I think that V and W could be computed faster
        MRS_merged= self.merge(MRS_new) # merge MRS_new
        MRS_new_rlens = [len(subdict) for subdict in MRS_merged] # number of conditions per rule
        MRS_new_len = len(MRS_new_rlens) # number of rules
        MRS_new_nterms = self.calculate_items(MRS_merged)
        W = np.zeros((MRS_new_len,self.N)) # matrix (number of rules)x(examples): 1 if an example satisfies a rule
        V = np.zeros((MRS_new_len,self.N)) # matrix (number of rules)x(examples): int with how many conditions each example satisfies in a rule 
        rm_attributes_list = [] # list of attributes to remove from model
        for i,subdict in enumerate(MRS_merged): # per rule in rule set
            for att in subdict.keys(): # per attribute in rule
                tmp = (np.sum(self.df[:,subdict[att]],axis = 1)>0).astype(int)
                # N vector saying if att is satisfied per individual (1) or not (0)
                V[i] += tmp # add 1 for each att satisfied
                if np.mean(tmp)==1: # if all examples verify 1 rule, remove attribute 
                    rm_attributes_list.append((i,att))
            W[i] = (V[i]==MRS_new_rlens[i]).astype(int) # 1 if the number of conditions per rule satisfied by a examples is the total length of such rule
        for i,att in rm_attributes_list: # remove those attributes that are verified by all samples
            MRS_merged[i].pop(att,None)
        Yhat = (np.sum(W,axis = 0)>0).astype(int) # predict 1 if at least one rule in the rule list is satisfied by individual
        TP,FP,TN,FN = getConfusion(Yhat,self.Y) # number (not percetage) of TP, FP, TN and FN
        Error = (FP + FN)/len(self.Y) # first term objective (accuracy)
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
            for row_indexA in range(len(MRS_merged)): # for each rule
                self.row_indexA = row_indexA # add index to model
                if row_indexA >= len(MRS_merged): # AFTER TONG MEETING
                    break
                attributeA = set(MRS_merged[row_indexA].keys()) # set of attributes in each rule
                if row_indexA not in rm_list: # if index row_index is not in rm_list
                    for row_indexB in range(len(MRS_merged)): # for another rule in the rule list different from the above
                        if row_indexB != row_indexA and row_indexB not in rm_list: # if another index is not also in rm_list
                            attributeB = set(MRS_merged[row_indexB].keys()) # set of attributes in each rule
                            if attributeA.issubset(attributeB): # if set of attributes A is a subset of the set of attributes B (recall that they where ordered according to length)
                                subset = 0 # initialization of (?)
                                same = 0 # initialization of (?)
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
                                    for attr in attributeA:
                                        MRS_merged[row_indexA][attr] = list(set(MRS_merged[row_indexA][attr]).union(set(MRS_merged[row_indexB][attr])))
                                    rm_list.append(row_indexB)           
                                elif subset == len(attributeA):  # delete attributeB directly because they are the same
                                    rm_list.append(row_indexB) 
                        row_indexB += 1
                    rm = []
                    for attr in MRS_merged[row_indexA].keys(): # per attribute per rule in MRS_merged
                        values = MRS_merged[row_indexA][attr] # values of such attribute
                        self.values = values # save values into model
                        self.MRS_merged = MRS_merged
                        if sum(np.sum(self.df[:,values],axis = 1)>0) == self.N:
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