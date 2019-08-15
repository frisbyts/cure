from sklearn.feature_selection import SelectKBest
from sklearn.metrics import accuracy_score, roc_curve, auc, f1_score, matthews_corrcoef, confusion_matrix, recall_score, classification_report
from sklearn.cross_validation import StratifiedKFold
from sklearn.base import BaseEstimator,ClassifierMixin,clone
from sklearn.model_selection import train_test_split,LeaveOneOut
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from scipy import interp
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt 
import sys 
#from MetaCost import MCost

def full_resamples(X, y, nresamples):
    n0 = np.sum(y == 0)  # do a stratified full resample
    n1 = np.sum(y == 1)
    _X = np.r_[X[y == 0], X[y == 1]]  # re-order to simplify things
    _y = np.r_[np.zeros(n0, int), np.ones(n1, int)]

    s = [None] * nresamples
    for i in range(nresamples):
        r0 = np.random.randint(0, n0, n0)  # full resample
        r1 = np.random.randint(0, n1, n1) + n0
        r = np.r_[r0, r1] 
        s[i] = (_X[r], _y[r])
    return s

class MCost(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator, C, use_predict_proba):
        super(MCost, self).__init__()
        self.base_estimator = base_estimator
        self.C = C 
        self.use_predict_proba = use_predict_proba
        self.ret = None
        self.classes_ = (0, 1)

    def fit(self, X, y): 
        X = np.asarray(X)
        y = np.array(y, copy=True)
        C = self.C

        # number of resamples to generate
        m = 10
        # number of examples in each resample
        # (can be smaller than dataset)
        n = len(X)/float(10)
        # do models produce class probabilities?
        p = self.use_predict_proba
        # are all resamples to be used for each example
        #q = True  # TODO: only True supported (recommended: False)

        # Step 1. Train everything
        M = [None] * m 
        for i, (Xt, yt) in enumerate(full_resamples(X, y, m)):
            m = clone(self.base_estimator)
            M[i] = m.fit(Xt, yt) 

        # Step 2. Per observation, action (i.e. relabel)
        for i in range(len(X)):  # observation
            if p:
                Pj = [m.predict_proba(X[[i]]) for m in M]
            else:
                Pj = [(1, 0) if m.predict(X[[i]]) == 0 else (0, 1) for m in M]
            P = np.mean(Pj, 0)
            j = np.argmax(P * C)
            y[i] = j

        # WEIRD: for whatever reason some models (LinearSVC for instance) need
        # more than one observation of each class to work
        if np.sum(y == 1) <= 1:
            self.ret = 0
        elif np.sum(y == 0) <= 1:
            self.ret = 1
        else:
            # Step 3. Train final model with new data
            self.model = clone(self.base_estimator)
            self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        if self.ret is None:
            return self.model.predict_proba(X)
        return np.zeros(len(X))

    def predict(self, X):
        if self.ret is None:
            return self.model.predict(X)
        return np.repeat(self.ret, len(X))

class OptimizedCost(BaseEstimator,ClassifierMixin):

    def __init__(self, base_estimator, metric_constrained, threshold, secondary_constraints):
        self.base_estimator = base_estimator
        self.metric_constrained = metric_constrained
        self.threshold = threshold
        self.secondary_constraints = secondary_constraints
       # self.roc_title = roc_title
       # self.score_png = score_png
       # self.sumscore_png = sumscore_png

    def fit(self, X, y):
        m = clone(self.base_estimator)
        return m.fit(X,y)

    def find_best_weights(self, X, y):
        con = self.metric_constrained
        sec = self.secondary_constraints.replace(" ", "").split(',')
        thresh = self.threshold
        base = self.base_estimator
        scores = ['sensitivity', 'specificity', 'auc', 'f1', 'matthews']
        if con not in scores:
            print("%s not supported.  Must be one of 'sensitivity', 'specificity', 'auc', 'f1', or 'matthews'" % con)
            sys.exit()
        second_indx = []
        for s in sec:
            if s not in scores:
                print("%s not supported.  Must be one of 'sensitivity', 'specificity', 'auc', 'f1', or 'matthews'" % s)
                sys.exit()
            second_indx.append(scores.index(s))
        indx = scores.index(con)

        outcomes = {}
        metrics = {0:sens, 1:spec, 2:roc_auc, 3:f1, 4:matt}
        weights = [1,1]
        test_ = 0
        #find where constraint greater than threshold
        print("Attaining primary constraint threshold. This will take some time")
        while test_ < thresh:
            if weights[1] > 50:
                initial_weight = [1,2]
                weights = [1,2]
                print("bad fold")
                break
            est1_ = MCost(base,weights,True)
            #print("LOO happening, current C: {}".format(weights))
            #loo goes here
            #needs to return:
            #prediction, predict_proba

            pred, probs = cross_validation(est1_,X,y) 

           #assume primary constraint not AUC
            if indx != 2: #auc takes different input
                test_ = metrics[indx](y,pred)
                if sec != 'auc':
                    sec_test_ = metrics[second_indx[0]](y,pred)
                else:
                    try:
                        sec_test_ = metrics[second_indx[0]](y,probs[:,1])
                        #print("auc: {}".format(sec_test_))
                    except:
                        sec_test_ = metrics[second_indx[0]](y,[1]*len(probs))
                        #print("auc: {}".format(sec_test_))
            else:
                try:
                    test_ = metrics[2](y,probs[:,1])
                except:
                    ppp = []
                    for j, n  in enumerate(probs):
                        if j % 2 == 1:
                            ppp.append(n) 
                    test_ = metrics[2](y,ppp)
            if test_ >= thresh:
                initial_weight = weights
                #print(confusion_matrix(y,pred))
                break
            weights = [1,weights[1]*2]

        best_pred = pred
        best_prob = probs
        #print(best_prob)
        #print("secondary constraint: {}".format(sec_test_))
        #optimize with secondary constraints 
        if weights == [1,1]:
            search_space = np.linspace(1,2,10)
        else:
            search_space = np.linspace(weights[1]/float(2),weights[1],10)
        #print(search_space)
        for w in search_space:
            outcomes[(1,w)] = np.zeros(5)
   
        possible_best_preds = []
        possible_best_probs = []
        
        print("Optimizing secondary constraint. This will also take some time")
        for i in search_space:
            est1 = MCost(base,[1,i],True)

            #loo goes here
            pred, probs = cross_validation(est1, X, y)
            possible_best_preds.append(pred)
            possible_best_probs.append(probs)
          
            for m in [indx]+second_indx:
                if m != 2: #auc takes different input
                    outcomes[(1,i)][m] = metrics[m](y,pred)
                else:
                    #print(metrics[2](y,probs[:,1]))
                    try:
                        outcomes[(1,i)][2] = metrics[2](y,probs[:,1])
                        #print("works")
                        #print(metrics[2](y,probs[:,1]))
                    except:
                        #print("issue")
                        pp = []
                        p = probs.ravel()
                        for j,n in enumerate(p):
                            if j % 2 == 1:
                                pp.append(n)        
                        outcomes[(1,i)][2] = metrics[2](y,pp)
                #print(outcomes[(1,i)])

        tot=0
        for i,key in enumerate(outcomes):
            #print("possible secondary constraint: {}".format(outcomes[key][second_indx[0]]))
            if outcomes[key][indx] >= thresh and outcomes[key][second_indx[0]] > sec_test_: #sum(outcomes[key]) > tot:
                sec_test_ = outcomes[key][second_indx[0]]
                #print("new secondary constraint value: {}".format(sec_test_))
                optimal_weight = list(key)
                best_pred = possible_best_preds[i]
                best_prob = possible_best_probs[i]

        try:
            #print(optimal_weight)
            print("predictions")
            for p in best_pred:
                print(p[0])
            print("probabilities")
            for q in best_prob:
                print(q[0][1])
        except:
            print("unable to improve secondary constraint")
            optimal_weight = initial_weight
            #print(optimal_weight)
            print("predictions")
            for p in best_pred:
                print(p[0])
            print("probabilities")
            for q in best_prob:
                print(q[0][1])

        learned_weights = optimal_weight
        final_spec = spec(y,best_pred[:,0])
        final_sens = sens(y,best_pred[:,0])
        final_acc = accuracy_score(y,best_pred[:,0])
        stats = (sec_test_,final_sens,final_spec,final_acc)
        print("Final stats (secondary constraint, sensitivity, specificity, accuracy):")
        print(stats)
        return stats, best_pred, best_prob

def spec(y_test,pred):
    cm = confusion_matrix(y_test, pred)
    return cm[0][0]/float(cm[0][0]+cm[0][1])

def sens(y_test,pred):
    cm = confusion_matrix(y_test, pred)
    return recall_score(y_test,pred)

def roc_auc(y_test,probs):
    fpr, tpr, thresholds = roc_curve(y_test, probs)
    return auc(fpr,tpr)

def f1(y_test,pred):
    return f1_score(y_test,pred)

def matt(y_test,pred):
    return matthews_corrcoef(y_test,pred) 

def cross_validation(model, X, y):
    loo = LeaveOneOut()
    cv = loo.split(X)
    predictions = []
    probas = []
    #try:
    #    select_X = SelectKBest(k=20).fit_transform(X,y)
    #except:
    #    select_X = X
    confuse_matrix = np.zeros((2,2))
    for i, (train, test) in enumerate(cv):
        sk_best = SelectKBest(k=20)
        select_X = sk_best.fit_transform(X[train],y[train])
        selection_mask = sk_best.get_support()
        #print("sample {}".format(test))
        #mf = model.fit(select_X[train],y[train])
        mf = model.fit(select_X,y[train])
        #ypred = mf.predict(select_X[test])
        Xtest = X[test]
        Xtest = Xtest[:,selection_mask]
        ypred = mf.predict(Xtest)
        #print(ypred)
        predictions.append(ypred)
        if ypred == 0 and y[test] == 0: #true negative
            confuse_matrix += np.asarray([[1,0],[0,0]])
        elif ypred == 1 and y[test] == 1: #true positive
            confuse_matrix += np.asarray([[0,0],[0,1]])
        else:
            confuse_matrix += confusion_matrix(y[test],ypred)
       # print("CM: {}".format(confuse_matrix))
        #print(classification_report(y[test],ypred))

        probas_ = mf.predict_proba(Xtest)
        #print("Testing")
        #print(probas_)
        try:
            if probas_ == 1 or probas_ == 0:
                #print("well well well")
                #print(X[test])
                if ypred[0] == 1:
                    probas_ = np.asarray([[0.0, 1.0]])
                elif ypred[0] == 0:
                    probas_ = np.asarray([[1.0, 0.0]])
                else:
                    print("how") 
        except:
            pass
        probas.append(probas_)
      
    #print("real CM : {}".format(confuse_matrix))
    #print("spec: {}".format(confuse_matrix[0][0]/float(confuse_matrix[0][0] + confuse_matrix[0][1])))
    #print("sens: {}".format(confuse_matrix[1][1]/float(confuse_matrix[1][1] + confuse_matrix[1][0])))
    #print("accuracy: {}".format((confuse_matrix[0][0] + confuse_matrix[1][1])/np.sum(confuse_matrix)))
    #print("testing")
    #print(predictions)
    #print(probas)
    return np.asarray(predictions), np.asarray(probas)

