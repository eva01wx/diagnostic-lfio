# Import generic packages
from __future__ import division
import pickle
import pandas as pd 
from math import isnan
import numpy as np
from math import lgamma
import itertools
from numpy.random import random
from random import sample
import time
from copy import deepcopy
import operator
from collections import Counter, defaultdict
from fim import fpgrowth,fim 
from scipy.sparse import csc_matrix
from bisect import *

# Function to convert a data set into a binary one:
# df: pandas data set
# collist: list of the names of continuous variables
# thresholds: dictionary with name & quantiles of each continuous variable
# --> df: binary pandas data set
def code_intervals_thresholds(df,collist,thresholds):
    for col in collist:
        for q in range(len(thresholds[col])-1):
            df[col+'_'+str(q)] = ((df[col] >=thresholds[col][int(q)]) & (df[col]<thresholds[col][int(q+1)])).astype(int).values
    df.drop(collist,axis = 1, inplace = True)

# Function to convert a raw data set into the binary format FastSRS expects.
# Each non-target column becomes one or more 0/1 indicator columns:
#   - binary:      `col_<v>`     (and `col_not<v>`  if include_negations)
#   - categorical: `col_<v>`     for each value v (and its negation)
#   - numerical:   `col_<=<t>`   for each quantile threshold t (and `col_><t>`)
# The target is appended as an int 'Class' column. Columns left unspecified are
# auto-detected by dtype: 2-unique-value -> binary, object/category -> categorical,
# otherwise -> numerical. Only '-' is replaced with '.' in the resulting column
# names (matching the existing datasets/*.csv files, several of which preserve
# spaces and parentheses in category values).
# data: pandas DataFrame or path to a CSV
# target_col: name of the class column in `data` (default 'Class')
# Nlevel: number of quantiles for numerical binarization (default 10)
# include_negations: also emit `_not<v>` / `_><t>` columns (default True)
# binary, categorical, numerical: optional explicit column-name overrides
# class_map: optional dict mapping raw class labels -> 0/1 (e.g. {' >50K': 1, ' <=50K': 0})
# missing_value, missing_label: optional pair to rename a category in the resulting
#     column names after encoding (e.g. ' ?' -> ' int' for adult.csv). The raw value
#     is left in place during encoding so it sorts naturally in np.unique.
# dropna: drop rows with NaN in any feature column before encoding (default True)
# output_path: if given, also write the result to this CSV path
# --> binarized pandas DataFrame with a final 'Class' column
def preprocess_data(data, target_col='Class', Nlevel=10, include_negations=True,
                    binary=None, categorical=None, numerical=None,
                    class_map=None, missing_value=None, missing_label=None,
                    dropna=True, output_path=None):
    if isinstance(data, str):
        df = pd.read_csv(data)
    else:
        df = data.copy()
    if dropna:
        df = df.dropna()

    if target_col not in df.columns:
        raise ValueError("target column '" + str(target_col) + "' not found in data")
    target = df[target_col]
    if class_map is not None:
        target = target.map(class_map)
    y = np.array(target, dtype=int)
    df = df.drop(target_col, axis=1)

    specified = set((binary or []) + (categorical or []) + (numerical or []))
    auto_binary, auto_categorical, auto_numerical = [], [], []
    for col in df.columns:
        if col in specified:
            continue
        if df[col].nunique() == 2:
            auto_binary.append(col)
        elif not pd.api.types.is_numeric_dtype(df[col]):
            # object / string (pandas >= 3 infers 'str') / category -> categorical
            auto_categorical.append(col)
        else:
            auto_numerical.append(col)
    binary = list(binary or []) + auto_binary
    categorical = list(categorical or []) + auto_categorical
    numerical = list(numerical or []) + auto_numerical

    # Binary: encode against just one value (the larger / last-sorted, matching
    # the existing FastSRS datasets: sex_1.0 in heart.csv, sex_ Male in adult.csv,
    # X1_1 in simulated_0.00.csv). The other value is implied by negation.
    for col in binary:
        values = np.unique(df[col])
        if len(values) < 1:
            continue
        val = values[-1]
        df[col + '_' + str(val)] = (df[col] == val).astype(int)
        if include_negations:
            df[col + '_not' + str(val)] = (df[col] != val).astype(int)
    df.drop(binary, axis=1, inplace=True)

    for col in categorical:
        for val in np.unique(df[col]):
            df[col + '_' + str(val)] = (df[col] == val).astype(int)
            if include_negations:
                df[col + '_not' + str(val)] = (df[col] != val).astype(int)
    df.drop(categorical, axis=1, inplace=True)

    thresholds = defaultdict(list)
    for col in numerical:
        thresholds[col] = np.unique(
            [df[col].quantile(float(q) / Nlevel) for q in range(1, Nlevel)]
        )
    for col in numerical:
        for t in thresholds[col]:
            # Cast to Python float so the column-name suffix is version-independent;
            # str(np.float64(x)) shortens in numpy 2.x but kept full precision in 1.x.
            tstr = str(float(t))
            df[col + '_<=' + tstr] = (df[col] <= t).astype(int).values
            if include_negations:
                df[col + '_>' + tstr] = (df[col] > t).astype(int).values
    df.drop(numerical, axis=1, inplace=True)

    df['Class'] = y
    df.columns = df.columns.str.replace('-', '.', regex=False)
    if missing_value is not None and missing_label is not None:
        df.columns = df.columns.str.replace(str(missing_value), str(missing_label),
                                            regex=False)

    if output_path is not None:
        df.to_csv(output_path, index=False)
    return df

# Function to extract rules from a tree coming from a RandomForest classifier
# tree: a tree coming from a RandomForest classifier
# feature_names: names of predictor variables in the data set    
def extract_rules(tree, feature_names):
    left      = tree.tree_.children_left
    right     = tree.tree_.children_right
    threshold = tree.tree_.threshold
    features  = [feature_names[i] for i in tree.tree_.feature]
    # get ids of child nodes
    idx = np.argwhere(left == -1)[:,0]     

    def recurse(left, right, child, lineage=None):          
        if lineage is None:
            lineage = []
        if child in left:
            parent = np.where(left == child)[0].item()
            suffix = '_neg'
        else:
            parent = np.where(right == child)[0].item()
            suffix = ''
        lineage.append((features[parent].strip()+suffix))

        if parent == 0:
            lineage.reverse()
            return lineage
        else:
            return recurse(left, right, parent, lineage)   
    rules = []
    for child in idx:
        rule = []
        for node in recurse(left, right, child):
            rule.append(node)
        rules.append(rule)
    return rules

# Function that iterates and accumulates elements in a list according to an operation
# iterable: list
# func: operation to be made
def accumulate(iterable, func=operator.add):
    'Return running totals'
    it = iter(iterable)
    total = next(it)
    yield total
    for element in it:
        total = func(total, element)
        yield total

# Function to calculate the number (not percentage) of TP, FP, TN and FN
# Yhat: N vector of predicted values for individuals
# Y: N vector of actual values of individuals
def getConfusion(Yhat,Y):
    if len(Yhat)!=len(Y): # checking if Yhat and Y refer to the same sample or not
        raise NameError('Yhat has different length')
    TP = np.dot(np.array(Y),np.array(Yhat))
    FP = np.sum(Yhat) - TP # number of total predicted positives - TP
    TN = len(Y) - np.sum(Y)-FP # number of total actual negatives - FP
    FN = len(Yhat) - np.sum(Yhat) - TN # number of total predicted negatives - TN
    return TP,FP,TN,FN

# Function to find the rightmost value in a less than x
# a: vector
# x: value
def find_lt(a, x):
    i = bisect_left(a, x)
    if i:
        return int(i-1)
    else:
        return 0

def predict_MRS(MRS,df):
    Z = np.zeros(len(df))
    for i,subdict in enumerate(MRS):
        tmp = np.zeros(len(df))
        for att in subdict.keys():
            tmp += (np.sum(df[:,subdict[att]],axis = 1)>0).astype(int)
        Z += (tmp == len(subdict)).astype(int)
    Yhat = (Z>0).astype(int)
    return Yhat

def predict_MRS_mix(MRS,df,itemNames,allthresholds,numericals):
    Z = np.zeros(len(df))
    for i,subdict in enumerate(MRS):
        tmp = np.zeros(len(df))
        for att in subdict.keys():
            if att in numericals:
                quantiles = [itemNames[item].split('_')[1] for item in subdict[att]]
                tmp += (np.sum([(np.array(df[att]>=allthresholds[att][int(q)])*np.array(df[att]<allthresholds[att][int(q)+1])).tolist() for q in quantiles],axis = 0)>0).astype(int)
            else:
                tmp += (np.sum(df[[itemNames[col] for col in subdict[att]]],axis = 1)>0).astype(int)
        Z += (tmp == len(subdict)).astype(int)
    Yhat = (Z>0).astype(int)
    return Yhat

def ROChull(points):
    from scipy.spatial import ConvexHull
    hull = ConvexHull(points)
    fpr = []
    tpr = []
    index = np.unique(hull.simplices)
    for i in index:
        if points[i][0]<=points[i][1]:
            fpr.append(points[i][0])
            tpr.append(points[i][1])
    fpr,tpr = zip(*sorted(zip(fpr,tpr)))
    return fpr,tpr

def code_intervals_nlevel(df,collist,Nlevel):
    for col in collist:
        for q in range(Nlevel):
            l = df[col].quantile(float(q)/Nlevel)
            u = df[col].quantile(float(q+1)/Nlevel)
            df[col+'_'+str(q)] = ((df[col] >=l) & (df[col]<u)).astype(int).values
    df.drop(collist,axis = 1, inplace = True)

def binary_code(df,collist,Nlevel):
    for q in range(1,Nlevel+1,1):
        thresholds = df[collist].quantile(float(q)/Nlevel)
        for col in collist:
            df[col+'_geq'+str(thresholds[col])] = (df[col] >= thresholds[col]).astype(float)
    df.drop(collist,axis = 1, inplace = True)


def code_categorical(df,colnames,missingvalue):
    for col in colnames:
        values = np.unique(df[col])
        for val in values:
            if val not in missingvalue:
                df[col+'_'+str(val)] = (df[col]==val).astype(int)
    df.drop(colnames, axis = 1, inplace = True)


def code_intervals_nlevel(df,collist,Nlevel):
    for col in collist:
        df[col+'_missing'] = isnan(df[col]).astype(int)
        index = np.where(~isnan(df[col]))
        for q in range(Nlevel):
            l = df[col].quantile(float(q)/Nlevel)
            u = df[col].quantile(float(q+1)/Nlevel)
            df[col+'_'+str(q)] = 0
            df[col+'_'+str(q)][index] = ((df[col][index] >=l) & (df[col][index]<u)).astype(int).values
    df.drop(collist,axis = 1, inplace = True) 

def merge_interval(model, grs):
    for i,subdict in enumerate(grs): # per rule in rule set and index
        for key in subdict.keys(): # per attr/condition in rule
            idx_ls = grs[i][key]
            values_ls=[]
            for value in subdict[key]:
                val=model.itemNames[value].split('_')
                val.pop(0)
                Cond='_'.join(val)
                values_ls.append(Cond)
            list_lw = []
            list_gt = []
            idx_list_lw = []
            idx_list_gt = []
            value_ls_final=[]
            idx_ls_final=[]
            idx_list_lw_new=[]
            idx_list_gt_new=[]
            idx_ls_new=[]
            for j in range(len(values_ls)):
                v_lw = values_ls[j].split('<=') 
                v_gt = values_ls[j].split('>') 
                if len(v_lw)==1 & len(v_gt)==1:
                    value_ls_final.append(v_lw[0])
                    idx_ls_final.append(idx_ls[j])
                if len(v_lw)>1:
                    list_lw.append(float(v_lw[1]))
                    idx_list_lw.append(idx_ls[j])
                
                if len(v_gt)>1:
                    list_gt.append(float(v_gt[1]))
                    idx_list_gt.append(idx_ls[j])
            if len(list_lw)>0 or len(list_gt)>0:
                list_lw=np.array(list_lw)
                idx_list_lw=np.array(idx_list_lw)
                list_gt=np.array(list_gt)
                idx_list_gt=np.array(idx_list_gt)
                if len(list_lw)>0:
                    idx_ls=np.argsort(-list_lw)
                    idx_list_lw_new=idx_list_lw[idx_ls]
                if len(list_gt)>0:
                    idx_ls=np.argsort(list_gt)
                    idx_list_gt_new=idx_list_gt[idx_ls]
                if len(idx_list_gt_new)==0:
                    idx_ls_new=[idx_list_lw_new[0]]
                if len(idx_list_lw_new)==0:
                    idx_ls_new=[idx_list_gt_new[0]]
                if len(idx_list_lw_new)!=0 &len(idx_list_gt_new)!=0:
                    idx_ls_new=[idx_list_lw_new[0]]+[idx_list_gt_new[0]]
                #idx_ls_new=[idx_list_lw_new[0]]+[idx_list_gt_new[0]]
            grs[i][key]=idx_ls_final+list(idx_ls_new)
def merge_logical(model, grs):
    for i,subdict in enumerate(grs): # per rule in rule set and index
        
        for key in subdict.keys():
            idx_ls = grs[i][key]
            values_ls=[]
            not_ls = []
            other_ls = []
            idx_ls_neg=[]
            idx_ls_pos=[]
            Cond_ls_neg=[]
            Cond_ls_pos=[]
            for value in subdict[key]:
                val=model.itemNames[value].split('_')
                val.pop(0)
                Cond='_'.join(val)
                values_ls.append(Cond)
           

            for j in range(len(values_ls)):
                #print("Cond",Cond)
                if 'not' in values_ls[j]:
                    idx_ls_neg.append(idx_ls[j])
                else:
                    idx_ls_pos.append(idx_ls[j])
          
            if len(idx_ls_neg)==0:
                grs[i][key]=idx_ls_pos
            else:
                grs[i][key]=idx_ls_neg
            
# Function to compute accuracy:
def accuracy(Yhat,Y):
    TP,FP,TN,FN = getConfusion(Yhat,Y)
    return float(TP+TN)/(TP+TN+FP+FN)

# Function to compute the number of rules of the rule set
def calculate_rules(MRS):
    return len(MRS)

# Function to compute the number of conditions of the rule set
def calculate_conditions(MRS): 
    return np.sum([len(MRS[j]) for j in range(len(MRS))])

# Function to compute the number of items of the rule set
def calculate_values(model,MRS):
    count_items = 0
    for i,subdict in enumerate(MRS):
        for key in subdict.keys():
            idx_ls = MRS[i][key]
            values_ls=[]
            for value in subdict[key]:
                val=model.itemNames[value].split('_')
                val.pop(0)
                Cond='_'.join(val)
                values_ls.append(Cond)
            count_items += len(values_ls)
    return count_items

def prediction_diversity(R1,R2,X):
    pred_R1=predict_MRS(R1,X)
    pred_R2=predict_MRS(R2,X)
    diversity_disagreement = np.sum(np.abs(np.asarray(pred_R1)-np.asarray(pred_R2)))/X.shape[0]
    return diversity_disagreement

def structural_diversity(R1, R2):
    cond_set_R1 = set()
    cond_set_R2 = set()
    for rule in R1:
        for c in rule.values():
            if len(c) > 0:
                # c can be a list or a condition object/string
                if isinstance(c, (list, tuple, set)):
                    for cc in c:
                        cond_set_R1.add(cc)
                else:
                    cond_set_R1.add(c)
    for rule in R2:
        for c in rule.values():
            if len(c) > 0:
                # c can be a list or a condition object/string
                if isinstance(c, (list, tuple, set)):
                    for cc in c:
                        cond_set_R2.add(cc)
                else:
                    cond_set_R2.add(c)
    union = len(cond_set_R1.union(cond_set_R2))

    intersection = len(cond_set_R1.intersection(cond_set_R2))
    jaccard = intersection / union

    diversity_R1_R2 = 1 - jaccard
    return diversity_R1_R2

def canonical_form(mrs):
    return tuple(
        sorted(
            tuple(
                (att, tuple(sorted(vals)))
                for att, vals in rule.items()
            )
            for rule in mrs
        )
    )


def get_epsilon_rashomon(Rset, eps):
    if len(Rset) == 0:
        # Train returned an empty Rashomon set (regularization too large).
        return []
    objectives = np.array([r[1] for r in Rset])

    best = objectives.min()

    idx = np.where(objectives <= (1 + eps) * best)[0]

    seen = set()
    unique_models = []

    for i in idx:
        model = Rset[i][0]
        key = canonical_form(model)

        if key not in seen:
            seen.add(key)
            unique_models.append(Rset[i])

    return unique_models
