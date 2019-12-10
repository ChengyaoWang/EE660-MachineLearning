# Official Library
import os
import numpy as np
import matplotlib.pyplot as plt
import statistics as stat
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, auc, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.svm import LinearSVC
from datetime import datetime
# Other Python Files
from utils import util
from dataGen import dataGenerator



class models():
    def __init__(self):
        os.system('clear')
        #Instantiate the data class
        self.data = dataGenerator(5, produceResult = False, forceRegen = False)
        self.util = util()
        print('-' * 100)
    
    def __str__(self):
        s = 'Models Covered:\n'
        s += 'Baseline: K Nearest Neighbor\n'
        s += 'BoostingTree & Boosting Forest\n'
        s += 'Elastic Net Logistic Regression\n'
        s += 'kMeans / Logistic Regression Combined\n'
        s += 'Explore Semi-supervised: Semi-supervised Support Vector Machine\n'
        s += 'Explore Unsupervised: Spectral Clustering\n'
        s += '-' * 100
        return s

    #Baseline: Vanilla KNN
    def baselineFit(self):
        print('Starting Baseline Model Fit using KNN = 5')
        knn_instance = KNeighborsClassifier(n_neighbors = 5,
                                            weights = 'uniform',
                                            metric = 'euclidean')
        print('Model is Vulerable to Train/Valid Splitting, Using CV = 5 to estimate validation Acc.')
        print('With Dataset After Vanilla PCA:')
        cv_zip_set, acc = self.util.input_CVdataset('Vani_CV_5'), []
        for test_valid_tuple in cv_zip_set:
            knn_instance.fit(test_valid_tuple[0], test_valid_tuple[2])
            acc.append(accuracy_score(knn_instance.predict(test_valid_tuple[1]), test_valid_tuple[3]))
        self.KNN_Vani_acc = stat.mean(acc)
        print(acc)
        print('Fit Complete: Accuracy Mean = {0:.5f}, Accuracy STD = {1:.5f}'.format(stat.mean(acc),
                                                                                     stat.stdev(acc)))
        print('With Dataset After Kernel PCA:')
        cv_zip_set, acc = self.util.input_CVdataset('Ker_CV_5'), []
        for test_valid_tuple in cv_zip_set:
            knn_instance.fit(test_valid_tuple[0], test_valid_tuple[2])
            acc.append(accuracy_score(knn_instance.predict(test_valid_tuple[1]), test_valid_tuple[3]))
        print(acc)
        print('Fit Complete: Accuracy Mean = {0:.5f}, Accuracy STD = {1:.5f}'.format(stat.mean(acc),
                                                                                     stat.stdev(acc)))
        self.KNN_Ker_acc = stat.mean(acc)
        print('-' * 100)
    def baselineTest(self):
        print('Starting Baseline Model Test using KNN = 5')
        X_train, X_test, y_train, y_test = self.util.input_dataset(datasetName = 'VaniPCA', returnType = 'list')
        knn_instance = KNeighborsClassifier(n_neighbors = 5,
                                            weights = 'uniform',
                                            metric = 'euclidean').fit(X_train, y_train)
        pred = knn_instance.predict(X_test)
        self.util.testReportGen('Baseline Model: K Nearest Neightbor', pred, y_test)
        print('-' * 100)
    '''
    Fine tune model for accuracy squeeze
    Using model: GradientBoosting Forest
    Subsampling instance is performed in Boosting Stage
    Subsampling feature is performed in Bagging Stage
    Hyperparameter List: name / default_val  
    Bagging:       1. n_estimators  / 10          -- Width of the forest
                   2. max_features  / 1.0         -- Subsampling features for weak learner
                   3. boostrap_features / False   -- Subsampling features with replacement for weak learner
    Boosting:      1. loss          / 'deviance', 'exponential'
                   2. learning_rate / 0.1         -- Learning rate of gradient descent
                   3. n_estimators  / 100         -- Depth of the forest (number of epochs)
                   4. subsample     / 1.0         -- Subsampling instances for each step
                   5. min_samples_split / 2       -- Will not split if < 2
                   6. min_samples_leaf  / 1       -- Have to split once
                   7. max_depth         / 3       -- Maximum depth of a weak learner
                   8. min_impurity increase  / 0  -- Will not split if < 0
                   9. max_leaf_nodes      / 3     -- Will not spilt > 
                   10.validation_fraction / 0.1   -- Fraction of training data aside for early stopping
                   11.n_iter_no_change    / None  -- Early stopping depending of the validation set
                   12.tol                 / 1e-4  -- Loss tolerance for early stoping above
    #Using Hierachical Cross Validation
    '''
    def modelInstantiate(self, cartPara, boostingPara, baggingPara):
        # These are rather fixed hyperparameters that don't Cross-Validate on
        # May be changed by hand.
        self.fixedParaBoost = {'loss': 'exponential'         
        }
        self.fixedParaCart = {'criterion': 'friedman_mse',
                              'max_leaf_nodes': None                # We have already Specified the Depth    
        }
        self.fixedParaBag = {'max_samples': 1.0,
                             'bootstrap_features': False,
                             'bootstrap': False}
        #These are the hyperparameters that Cross-Validate on
        #Discrete hyperparameter space, use hierachical grid search
        max_depth, max_features, min_samples_split, min_samples_leaf = cartPara
        n_estimators_boost, subsample, learning_rate = boostingPara
        n_estimators_bag, max_features = baggingPara

        boostingTree = GradientBoostingClassifier(#Parameters for Boosting Framework
                                                  loss = self.fixedParaBoost['loss'],
                                                  learning_rate = learning_rate,
                                                  n_estimators = n_estimators_boost,
                                                  subsample = subsample,
                                                  #Parameters for Weak learners: CARTs
                                                  criterion = self.fixedParaCart['criterion'],
                                                  min_samples_split = min_samples_split,
                                                  min_samples_leaf = min_samples_leaf,
                                                  min_weight_fraction_leaf = 0,
                                                  max_depth = max_depth,
                                                  min_impurity_decrease = 0,
                                                  max_features = max_features,
                                                  max_leaf_nodes = self.fixedParaCart['max_leaf_nodes'],
                                                  #Parameters for Early Stopping
                                                  validation_fraction = 0.1,
                                                  n_iter_no_change = None,
                                                  tol = 1e-4,
                                                  #Other parameters
                                                  verbose = 0)
        forestModel = BaggingClassifier(boostingTree,
                                        n_estimators = n_estimators_bag,
                                        max_features = max_features,
                                        bootstrap_features = self.fixedParaBag['bootstrap_features'],
                                        max_samples = self.fixedParaBag['max_samples'],
                                        bootstrap = self.fixedParaBag['bootstrap'],
                                        oob_score = False)
        return boostingTree, forestModel
    def AccSqueeze_tree_toy(self):
        print('Performing untuned BoostingForest using initial parameters')
        cv_zip_set, acc = self.data.utils.input_CVdataset('Vani_CV_5'), []
        _, model = self.modelInstantiate([1, 'sqrt', 2, 1],
                                         [600, 1.0, 0.1],
                                         [50, 1.0])
        for X_train, X_valid, y_train, y_valid in cv_zip_set:
            model.fit(X_train, y_train)
            acc_cur = accuracy_score(model.predict(X_valid), y_valid)
            acc.append(acc_cur)
        print('CV Performance of Untuned BoostingForest:')
        print('{2}\nMean_Acc {0:.4f}, Std_Acc {1:.4f}'.format(stat.mean(acc), stat.stdev(acc), acc))
        print('-' * 100)
    def modelValidate(self, model, cv_zip_set):
        accList = []
        for X_train, X_valid, y_train, y_valid in cv_zip_set:
            model.fit(X_train, y_train)
            accList.append(accuracy_score(model.predict(X_valid), y_valid))
        return stat.mean(accList), stat.stdev(accList), accList
    def fetchNewest(self, parameter, stageNum):
        if stageNum >= 1:
            parameter['boost'][0] = 100
            parameter['boost'][2] = 0.1
        if stageNum >= 2:
            parameter['cart'][0] = 5
        if stageNum >= 3:
            parameter['cart'][3] = 6
            parameter['cart'][2] = 10
        if stageNum >= 4:
            parameter['cart'][1] = 8
        if stageNum >= 5:
            parameter['boost'][0] = 400
            parameter['boost'][2] = 0.025
        if stageNum >= 6:
            parameter['bag'][0] = 50
            parameter['bag'][1] = 15
        return parameter
    def AccSqueeze_tree_main(self, stage, showBest = False, verbose = 0):
        print('Model selection via Hierachical Cross-Validation')
        # Parameter initialization
        hyperparameter = {'cart':  [3, 'sqrt', 20, 5],               #Initial parameter for CART
                          'boost': [60, 0.8, 0.1],                   #Initial parameter for Boosting
                          'bag':   [50, 0.9],                        #Initial parameter for Bagging
                          'dummy': [1]}
        cv_zip_set = self.data.utils.input_CVdataset('Vani_CV_5')
        # Coarse tuning Boosting parameter, be ready for CART tuning
        # Stage1:   Coarse Tune Boosting Parameter  : Boost_Depth + lr
        # Stage2_1: Fine Tune CART                  : CART_Depth + min_samples_spilt
        # Stage2_2: Fine Tune CART                  : min_samples_spilt + min_samples_leaf
        # Stage2_3: Fine Tune CART                  : max_features
        # Stage3:   Fine Tune Boosting Parameter    : Boost_Depth + lr
        # Stage4:   Fine Tune Bagging Parameter     : Bag_Width + max_features
        titleStr = ['Coarse Tune for Boosting: lr & BoostDepth',
                    'Fine Tune CART: CARTDepth & min_samples_spilt',
                    'Fine Tune CART: min_samples_spilt & min_samples_leaf',
                    'Fine Tune CART: max_features',
                    'Fine Tune Boosting: lr & BoostDepth',
                    'Fine Tune Bagging: BagWidth & max_features']
        verboseStr = ['lr = {0}, Depth = {1}, Acc_mean = {2:0.5f}, Acc_std = {3:0.5f}',
                      'max_depth = {0}, min_samples_split = {1}, Acc_mean = {2:0.5f}, Acc_std = {3:0.5f}',
                      'min_samples_leaf = {0}, min_samples_split = {1}, Acc_mean = {2:0.5f}, Acc_std = {3:0.5f}',
                      'max_features = {1:.3f} Acc_mean = {2:0.5f}, Acc_std = {3:0.5f}',
                      'lr = {0:.3f}, Depth = {1:.3f}, Acc_mean = {2:0.5f}, Acc_std = {3:0.5f}',
                      'Width = {0}, Max_features = {1}, Acc_mean = {2:0.5f}, Acc_std = {3:0.5f}']
        CV_range = [[[0.05, 0.1, 0.2], [5 * x for x in range(1, 41)]],
                    [[x + 1 for x in range(0, 6)], [2 * (x + 1) for x in range(0, 21)]],
                    [[x + 2 for x in range(0, 7)], [2 * (x + 1) for x in range(0, 21)]],
                    [[1], [(x + 1) for x in range(0, 17)]],
                    [[0.3, 0.2, 0.15, 0.10, 0.05, 0.025, 0.01, 0.005], [int(100 * 1.05**x) for x in range(0, 60)]],
                    [[10 * x for x in range(1, 6)], [x + 1 for x in range(0, 17)]]]
        keyWord = [[['boost', 2], ['boost', 0]],                  #lr, BoostDepth
                   [['cart', 0], ['cart', 2]],                    #CART_depth, min_samples_spilt
                   [['cart', 2], ['cart', 3]],                    #min_samples_spilt, min_samples_leaf
                   [['dummy', 0], ['cart', 1]],                   #Max Features
                   [['boost', 2], ['boost', 0]],                  #lr, BoostDepth
                   [['bag', 0], ['bag', 1]]]                      #Bag_width, max_features
        filename = ['Coarse Tune Boosting',
                    'Fine Tune CART 1',
                    'Fine Tune CART 2',
                    'Fine Tune CART 3',
                    'Fine Tune Boosting',
                    'Fine Tune Bagging']
        axisLabel = [['Learning Rate', 'Boosting Depth'],
                     ['CART Depth', 'min_samples_split'],
                     ['min_samples_split', 'min_samples_leaf'],
                     [None, 'Max Features'],
                     ['Learning Rate', 'Boosting Depth'],
                     ['BagWidth', 'Max Features']]
        currentStage = -1
        for stagePnt in stage:
            currentStage += 1
            if not stagePnt:
                print('{0} Skipped'.format(titleStr[currentStage]))
            if stagePnt:
                print(titleStr[currentStage])
                best_record, acc = [], []
                keyWord_X, index_X = keyWord[currentStage][0][0], keyWord[currentStage][0][1]
                keyWord_y, index_y = keyWord[currentStage][1][0], keyWord[currentStage][1][1]
                hyperparameter = self.fetchNewest(hyperparameter, currentStage)
                progressCnt, totalStep = 0, len(CV_range[currentStage][0]) * len(CV_range[currentStage][1])
                startTime = datetime.now()
                for x in CV_range[currentStage][0]:
                    acc_temp = []
                    for y in CV_range[currentStage][1]:
                        hyperparameter[keyWord_X][index_X] = x
                        hyperparameter[keyWord_y][index_y] = y
                        if currentStage != 6:
                            model, _ = self.modelInstantiate(hyperparameter['cart'],
                                                             hyperparameter['boost'],
                                                             hyperparameter['bag'])
                        elif currentStage == 5:
                            _, model = self.modelInstantiate(hyperparameter['cart'],
                                                             hyperparameter['boost'],
                                                             hyperparameter['bag'])
                        mean, std, _ = self.modelValidate(model = model, cv_zip_set = cv_zip_set)
                        if mean >= 0.965:               best_record.append([x, y, mean])
                        acc_temp.append(mean)
                        if verbose == 0:            print('Progress: {0}/{1}'.format(progressCnt, totalStep), end = '\r')
                        elif verbose == 1:          print(verboseStr[currentStage].format(x, y, mean, std))
                        progressCnt += 1
                    acc.append(acc_temp)
                print('Completed: {0}\nTime Elapsed: {1}'.format(filename[currentStage], datetime.now() - startTime))
                if showBest:
                    print('Cases where acc > 96.5%:')
                    for token in best_record:
                        print(token)
                xRange = [token for token in CV_range[currentStage][1]]
                legendList = [str(token) for token in CV_range[currentStage][0]]
                self.util.Tree_plotting(resultList = acc, xRange = xRange,
                                        xLabel = axisLabel[currentStage][1], yLim = (0.93, 1),
                                        legendList = legendList, legendTitle = axisLabel[currentStage][0],
                                        filename = filename[currentStage], plot = False)
        print('-' * 100)
    def AccSqueeze_tree_last(self):
        hyperparameter = {'cart':  [3, 'sqrt', 20, 5],
                          'boost': [60, 0.8, 0.1],
                          'bag':   [50, 0.9],
                          'dummy': [1]}
        hyperparameter = self.fetchNewest(hyperparameter, 10)
        cv_zip_set = self.util.input_CVdataset('Vani_CV_5')
        boostTree, model = self.modelInstantiate(hyperparameter['cart'], hyperparameter['boost'], hyperparameter['bag'])
        mean1, std1, accList1 = self.modelValidate(boostTree, cv_zip_set = cv_zip_set)
        mean2, std2, accList2 = self.modelValidate(model, cv_zip_set = cv_zip_set)
        print('Final CV Performance of BoostingTree:')
        print('Mean_Acc {0:.4f}, Std_Acc {1:.4f}'.format(mean1, std1), '\n', accList1)
        print('Final CV Performance of BoostingForest:')
        print('Mean_Acc {0:.4f}, Std_Acc {1:.4f}'.format(mean2, std2), '\n', accList2)
        print('-' * 100)
    def AccSqueeze_tree_test(self):
        print('Starting tree based model Test: Boosting Tree & Boosting Forest')
        hyperparameter = {'cart':  [3, 'sqrt', 20, 5],
                          'boost': [60, 0.8, 0.1],
                          'bag':   [50, 0.9],
                          'dummy': [1]}
        hyperparameter = self.fetchNewest(hyperparameter, 10)
        X_train, X_test, y_train, y_test = self.util.input_dataset(datasetName = 'VaniPCA', returnType = 'list')
        boostTree, boostForest = self.modelInstantiate(hyperparameter['cart'],
                                                       hyperparameter['boost'],
                                                       hyperparameter['bag'])
        boostTree.fit(X_train, y_train)
        boostForest.fit(X_train, y_train)
        pred_tree = boostTree.predict(X_test)
        pred_forest = boostTree.predict(X_test)
        self.util.testReportGen('Tree Based Model: AdaBoosting Tree', pred_tree, y_test)
        self.util.testReportGen('Tree Based Model: AdaBoosting Forest', pred_forest, y_test)
        print('-' * 100)
    '''
    # Fine tune model for accuracy squeeze
    # Using model: Logistic Regression
    # Hyperparameters: penaltyType: l1 + l2 (Elastic Net), penalty Str: [1e-8, 1e-7, ......, 1e7, 1e8]
    # Use Grid Search Cross Validation
    '''
    def modelInstantiate_Linear(self, C, l1_ratio):
        lr_instance = LogisticRegression(penalty = 'elasticnet',
                                         solver = 'saga',
                                         C = C,
                                         max_iter = 10000,
                                         l1_ratio = l1_ratio)
        return lr_instance
    def modelValidate_Linear(self, model, cv_zip_set):
        accList = []
        for X_train, X_valid, y_train, y_valid in cv_zip_set:
            model.fit(X_train, y_train)
            accList.append(accuracy_score(model.predict(X_valid), y_valid))
        return stat.mean(accList), stat.stdev(accList), accList
    def AccSqueeze_linear_main(self, verbose = 0):
        print('Fine Tuning Logistic Regression: Penalty Strength + Elastic Net Ratio')
        cv_zip_set = self.data.utils.input_CVdataset('Vani_CV_5')
        regRange, ratioRange = [x for x in range(-8, 9)], [0.2 * x for x in range(0, 6)]
        acc, verboseFrame = [], 'logRegStr = {0:.4f}, ElasticRatio = {1:.4f}, Mean = {2:.4f}, Std = {3:.4f}'
        progressCnt, totalStep = 0, len(regRange) * len(ratioRange)
        startTime = datetime.now()
        for ratio in ratioRange:
            acc_temp = []
            for C in regRange:
                model = self.modelInstantiate_Linear(np.exp(-C), ratio)
                mean, std, accList = self.modelValidate_Linear(model, cv_zip_set)
                acc_temp.append(mean)
                if   verbose == 0:         print('Progress: {0}/{1}'.format(progressCnt, totalStep), end = '\r')
                elif verbose == 1:         print(verboseFrame.format(C, ratio, mean, std))
                progressCnt += 1
            acc.append(acc_temp)
        print('Completed: {0}\nTime Elapsed: {1}'.format('Elastic Net Logistic Regression',
                                                         datetime.now() - startTime))
        #Plot the result
        self.util.Linear_plotting(resultList = acc,
                                  xRange = regRange,
                                  xLabel = 'log - Regularization Strength',
                                  yLim = (0.70, 0.99),
                                  legendList = ['{0:.1f}'.format(token) for token in ratioRange],
                                  legendTitle = 'ElasticNet Ratio',
                                  filename = 'Fine Tune Logistic Regression',
                                  plot = False)
        print('-' * 100)
    def AccSqueeze_linear_last(self):
        cv_zip_set, para = self.util.input_CVdataset('Vani_CV_5'), [np.exp(-1), 0.2]
        model = self.modelInstantiate_Linear(para[0], para[1])
        mean, std, accList = self.modelValidate_Linear(model, cv_zip_set = cv_zip_set)
        print('Final CV Performance of ElasticNet Logistic Regression:')
        print('{2}\nMean_Acc {0:.4f}, Std_Acc {1:.4f}'.format(mean, std, accList))
        print('-' * 100)
    def AccSqueeze_linear_test(self):
        print('Starting linear model Test: Elastic Net Logistic Regression')
        X_train, X_test, y_train, y_test = self.util.input_dataset(datasetName = 'VaniPCA', returnType = 'list')
        model = self.modelInstantiate_Linear(np.exp(-1), 0.2)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        self.util.testReportGen('Linear Model: Elastic Net Logistic Regression', pred, y_test)
        print('-' * 100)
    '''
    # Fine Tune model for accuracy Squeeze
    # Using model: K-means Clustering + Logistic Regression
    # Hyperparameters: Penalty Str, K
    # Use Hierachical Cross Validation
    # Obviously, increasing the number of clusters will result in overfit.
    # But it's hard to include this hyperparameter into cross validation
    # Select using plotting & kneel points
    '''
    def modelInstantiate_kMeans(self, kmeansPara, lrPara):
        n_clusters, C = kmeansPara, lrPara
        kmeans = KMeans(n_clusters = n_clusters,
                        n_init = 50,
                        max_iter = 10000)
        lrList = []
        for iter in range(kmeansPara):
            lr = LogisticRegression(penalty = 'elasticnet',
                                    solver = 'saga',
                                    C = np.exp(-C),
                                    max_iter = 10000,
                                    l1_ratio = 0.2)
            lrList.append(lr)
        return kmeans, lrList
    def AccSqueeze_kMeans_kCV(self, X, y):
        kRange, score = [x for x in range(2, 101)], []
        for k in kRange:
            model, _ = self.modelInstantiate_kMeans(k, 0)
            model.fit(X)
            score.append(np.log(-model.score(X)))
        plt.plot(kRange, score)
        plt.title('Log(Loss) - k')
        plt.savefig('../Results/Loss K Means.png', dpi = 400)
    def modelValidate_kMeans(self, cv_zip_set, para, test = False):
        kmeans, LogRegList = self.modelInstantiate_kMeans(para[0], para[1])
        accList = []
        for X_train, X_valid, y_train, y_valid in cv_zip_set:
            ClusteredTrain = [[[], []] for _ in range(para[0])]
            # Use K-Means to Cluster X_Train
            kMeans_pred = kmeans.fit_predict(X_train).tolist()
            for instancePnt in range(len(kMeans_pred)):
                ClusteredTrain[kMeans_pred[instancePnt]][0].append(X_train[instancePnt])
                ClusteredTrain[kMeans_pred[instancePnt]][1].append(y_train[instancePnt])
            # Sklearn Logistic Regression model can't train on identical label datasets
            # In such cases, kmeans predict on behalf of LR models
            # lrModelMask = None if labels not pure; 0 if all 0; 1 if all 1
            # Train each Cluster
            lrModelMask = [None for _ in range(para[0])]
            for XytuplePnt in range(len(ClusteredTrain)):
                X, y = ClusteredTrain[XytuplePnt][0], ClusteredTrain[XytuplePnt][1]
                purity = 1.0 * sum(y) / len(y)
                if purity == 0 or purity == 1:
                    lrModelMask[XytuplePnt] = purity
                    continue
                LogRegList[ClusteredTrain.index([X, y])].fit(X, y)
            # Make Predictions
            pred, acc_path = [], []
            for X, y in zip(X_valid, y_valid):
                kmeans_label = kmeans.predict([X])[0]
                if lrModelMask[kmeans_label] == 1 or lrModelMask[kmeans_label] == 0:
                    lr_label = lrModelMask[kmeans_label]
                else:
                    lr_label = LogRegList[kmeans_label].predict([X])[0]
                pred.append(lr_label)
                acc_path.append(int(lr_label == y))
            accList.append(1.0 * sum(acc_path) / len(acc_path))
            if test:            return pred
        return stat.mean(accList), stat.stdev(accList), accList
    def AccSqueeze_kMeans_main(self, verbose = 1):
        print('Fine Tuning Kmeans + LogRegression: k + Penalty Strength')
        cv_zip_set = self.util.input_CVdataset('Vani_CV_5')
        # Whole dataset
        X, y = cv_zip_set[0][0] + cv_zip_set[0][1], cv_zip_set[0][2] + cv_zip_set[0][3]
        # First try unsupervised method k - score
        #self.AccSqueeze_kMeans_kCV(X, y)
        # Kneel point found: K = 20, We inherit Elastic Net ratio 0.2 from previous results
        kRange, regRange, acc = [2, 5, 10, 15, 20, 40], [x for x in range(-9, 10)], []
        verboseFrame = 'logRegStr = {1:.4f}, Number of Clusters = {0}, Mean = {2:.4f}, Std = {3:.4f}'
        progressCnt, totalStep = 0, len(kRange) * len(regRange)
        startTime = datetime.now()
        for k in kRange:
            acc_temp = []
            for reg in regRange:
                mean, std, accList = self.modelValidate_kMeans(cv_zip_set, [k, reg], False)
                acc_temp.append(mean)
                if   verbose == 0:         print('Progress: {0}/{1}'.format(progressCnt, totalStep), end = '\r')
                elif verbose == 1:         print(verboseFrame.format(k, reg, mean, std))
                progressCnt += 1
            acc.append(acc_temp)
        print('Completed: {0}\nTime Elapsed: {1}'.format('KMeans preprocessed Logistic Regression',
                                                         datetime.now() - startTime))
        # Plot the result
        self.util.kMeans_plotting(resultList = acc,
                                  xRange = regRange,
                                  yLim = (0.90, 1),
                                  legendList = [str(k) for k in kRange],
                                  legendTitle = 'Num of Clusters',
                                  filename = 'kMeans-LogReg',
                                  plot = False)
        print('-' * 100)
    def AccSqueeze_kMeans_last(self):
        cv_zip_set, para = self.util.input_CVdataset('Vani_CV_5'), [2, 0]
        mean, std, accList = self.modelValidate_kMeans(cv_zip_set = cv_zip_set, para = para, test = False)
        print('Final CV Performance of kMeans Preprocessed logRegresion:')
        print('{2}\nMean_Acc {0:.4f}, Std_Acc {1:.4f}'.format(mean, std, accList))
        print('-' * 100)
    def AccSqueeze_kMeans_test(self):
        print('Starting kMeans Preprocessed logRegresion Test:')
        X_train, X_test, y_train, y_test = self.util.input_dataset(datasetName = 'VaniPCA', returnType = 'list')
        test_zip_set, para = [[X_train, X_test, y_train, y_test]], [2, 0]
        pred = self.modelValidate_kMeans(cv_zip_set = test_zip_set, para = para, test = True)
        self.util.testReportGen('Linear Model: Elastic Net Logistic Regression', pred, y_test)
        print('-' * 100)
    '''
    # Predicting using semi-supervised method
    # Using model: l1 - SVM
    # Hyperparameters: penalty-Str
    # Monte - Carlo Method M = 10
    '''
    def modelInstantiate_SVM(self, C):
        model = LinearSVC(penalty = 'l1',
                          loss = 'squared_hinge',
                          dual = False,
                          C = C,
                          max_iter = 500000)
        return model
    def Explore_SSSVM_CV(self, dataset, PenStrRange, fold):
        accList, splitList = [], StratifiedKFold(n_splits = fold, shuffle = True).split(dataset[0], dataset[1])
        X, y = np.asarray(dataset[0]), np.asarray(dataset[1])
        for penalty in PenStrRange:
            model, acc = self.modelInstantiate_SVM(np.exp(-penalty)), 0.0
            #Cross Validation
            for train_index, valid_index in splitList:
                model.fit(X[train_index], y[train_index])
                acc += accuracy_score(model.predict(X[valid_index]), y[valid_index])
            accList.append(acc / fold)
        Cbest = PenStrRange[accList.index(max(accList))]
        # Augmented y_train by y_train;
        #   (true, predicted) for Unlabeled
        #   (true, true) for Labeled
        y_train_aug = [[token, token] for token in dataset[1]]
        return self.modelInstantiate_SVM(np.exp(-Cbest)), y_train_aug
    def Explore_SSSVM_oneIter(self, model, zip_dataset):
        X_unlabeled, X_labeled, y_unlabeled, y_labeled = zip_dataset
        model.fit(X_labeled, [token[1] for token in y_labeled])
        score = model.decision_function(X_unlabeled).tolist()
        # Most Confident Instance
        tokenIndex = score.index(max(score, key = abs))
        pred_label = model.predict([X_unlabeled[tokenIndex]])
        X_labeled.append(X_unlabeled[tokenIndex])
        #Need modification
        OneResult = (y_unlabeled[tokenIndex] == pred_label)
        y_labeled.append([y_unlabeled[tokenIndex], pred_label[0]])
        del X_unlabeled[tokenIndex]
        del y_unlabeled[tokenIndex]
        return OneResult, [X_unlabeled, X_labeled, y_unlabeled, y_labeled]
    def Explore_SSSVM_do(self, zip_dataset, PenStrRange):
        X_unlabeled, X_labeled, y_unlabeled, y_labeled = zip_dataset
        pred_path = []
        model, y_labeled = self.Explore_SSSVM_CV([X_labeled, y_labeled], PenStrRange, 5)
        zip_dataset = [X_unlabeled, X_labeled, y_unlabeled, y_labeled]
        while zip_dataset[0]:
            #OneResult: True : correct / False : incorrect
            OneResult, zip_dataset = self.Explore_SSSVM_oneIter(model, zip_dataset)
            pred_path.append(int(OneResult))
        #Transform pred_path into accuracy
        temp = []
        for iter in range(len(pred_path)):
            temp.append(sum(pred_path[:iter + 1]) / float(iter + 1))
        # Return final fitted model
        model.fit(zip_dataset[1], [token[1] for token in zip_dataset[3]])
        return temp, model
    def Explore_SSSVM_main(self, repeat = 50):
        print('Implementing Semi-supervised Learning: l1 - SVM')
        acc_record, PenStrRange, ratioRange = [], [x for x in range(-8, 9)], [0.2, 0.4, 0.6, 0.8]
        verboseFrame = 'Labeled Ratio: {0}, Progress: {1} / {2}'
        for ratio in ratioRange:
            startTime = datetime.now()
            acc_temp = []
            for iter in range(repeat):
                #Spilt the Labled Data and Unlabeled Data
                zip_dataset = self.util.input_SSSVMdataset(ratio)
                acc_path, _ = self.Explore_SSSVM_do(zip_dataset, PenStrRange)
                acc_temp.append(acc_path)
                print(verboseFrame.format(ratio, iter + 1, repeat), end = '\r')
            print('Ratio = {0:.2f} Completed, Time Elapsed: {1}'.format(ratio, datetime.now() - startTime))
            acc_record.append(acc_temp)
        self.data.utils.SSSVM_plotting(acc_record = acc_record,
                                       xRange = [[x for x in range(len(acc_record[iter][0]))] for iter in range(len(acc_record))],
                                       yLim = (0.70, 1),
                                       filename = 'SSSVM-MonteCarlo',
                                       plot = False)
        # Report the final results
        print('Final Statistics for labeled ratio: 0.2, 0.4, 0.6, 0.8')
        for ratio, ratio_row in zip(ratioRange, acc_record):
            finalResult_temp = [row[-1] for row in ratio_row]
            print('Ratio = {0:.1f}, Mean = {1:.4f}, Std = {2:.4f}'.format(ratio,
                                                                          stat.mean(finalResult_temp),
                                                                          stat.stdev(finalResult_temp)))
        print('-' * 100)
    def Explore_SSSVM_test(self):
        print('Starting SSSVM Inductive Test:')
        X_train, X_test, y_train, y_test = self.util.input_dataset(datasetName = 'VaniPCA', returnType = 'list')
        # Split the training set into 1:1 labeled / unlabeled
        zip_dataset, PenStrRange = self.util.input_SSSVMdataset(0.5), [x for x in range(-8, 9)]
        acc_path, model = self.Explore_SSSVM_do(zip_dataset = zip_dataset, PenStrRange = PenStrRange)
        pred = model.predict(X_test)
        self.util.testReportGen('Semi-supervised Learning: SSSVM', pred, y_test)
        print('-' * 100)
    '''
    # Predicting using unsupervised method
    # Using model: Spectral Clustering
    # Hyperparameters: Kernel Type + Kernel Parameter
    # Cross Validation
    '''
    def modelInstantiate_SpeCluster(self, para):
        affinity, x = para
        model = SpectralClustering(n_clusters = 2,
                                   n_init = 50, 
                                   affinity = affinity,
                                   assign_labels = 'kmeans')
        if para[0] in ['rbf', 'laplacian']:
            model.set_params(gamma = np.exp(x))
        elif para[0] == 'polynomial':
            model.set_params(degree = x, gamma = 0.01, coef0 = 1000)
        elif para[0] == 'nearest_neighbors':
            model.set_params(n_neighbors = x)
        return model
    def modelValidate_SpeCluster(self, model, cv_zip_set):
        for X_train, X_valid, y_train, y_valid in cv_zip_set:
            model.fit(X_train + X_valid)
            mean = accuracy_score(model.labels_, y_train + y_valid)
            break
        return mean
    def Explore_SpeCluster_main(self, verbose = 0):
        print('Fine Tuning Spectral Clustering: Kernel + para')
        cv_zip_set = self.data.utils.input_CVdataset('Vani_CV_5')
        cv_para = [['rbf', 'gamma', [x for x in range(-10, 3)]],
                   ['laplacian', 'gamma', [x for x in range(-10, 3)]],
                   ['polynomial', 'degree', [x for x in range(1, 14)]],
                   ['nearest_neighbors', 'n_neighbor', [x for x in range(1, 14)]]]
        acc, verboseFrame = [], 'Kernel: {0}, Parameter: {1}, Value: {2:.4f}, Mean = {3:.4f}'
        progressCnt, totalStep = 0, sum([len(cv_para[iter][2]) for iter in range(len(cv_para))])
        startTime = datetime.now()
        for cv_para_set in cv_para:
            kernelType, paraName, para_range = cv_para_set
            acc_temp = []
            for para in para_range:
                model = self.modelInstantiate_SpeCluster([kernelType, para])
                mean = self.modelValidate_SpeCluster(model, cv_zip_set)
                if mean < 0.5:          mean = 1 - mean
                acc_temp.append(mean)
                if verbose == 0:        print('Progress: {0}/{1}'.format(progressCnt, totalStep), end = '\r')
                elif verbose == 1:      print(verboseFrame.format(kernelType, paraName, para, mean))
                progressCnt += 1
            acc.append(acc_temp)
        print('Completed: {0}\nTime Elapsed: {1}'.format('Elastic Net Logistic Regression',
                                                         datetime.now() - startTime))
        #Plot the result
        self.data.utils.SpeCluster_plotting(resultList = acc,
                                            xRange = [row[2] for row in cv_para],
                                            yLim = (0.50, 1),
                                            legendList = [row[0:2] for row in cv_para],
                                            filename = 'Fine Tune Spectral Clustering',
                                            plot = False)
        print('-' * 100)
    def Explore_SpeCluster_last(self):
        cv_zip_set = self.data.utils.input_CVdataset('Vani_CV_5')
        model = self.modelInstantiate_SpeCluster(['nearest_neighbors', 10])
        mean = self.modelValidate_SpeCluster(model, cv_zip_set = cv_zip_set)
        if mean < 0.5:          mean = 1 - mean
        print('Final Performance of Spectral Clustering:')
        print('Mean_Acc {0:.4f}'.format(mean))
        print('-' * 100)
    def Explore_SpeCluster_test(self):
        print('Starting Unsupervised model Test: Spectral Clustering')
        X_train, X_test, y_train, y_test = self.util.input_dataset(datasetName = 'VaniPCA', returnType = 'list')
        y_test = [token[0] for token in y_test]
        model = self.modelInstantiate_SpeCluster(['nearest_neighbors', 10])
        model.fit(X_train + X_test)
        pred, trueLabel = model.labels_, y_train + y_test
        mean = accuracy_score(pred, trueLabel)
        # See if flipping labels are needed
        if mean < 0.5:
            for tokenPnt in range(len(pred)):
                pred[tokenPnt] = int(pred[tokenPnt] == 0)
        self.util.testReportGen('Unsupervised Model: Spectral Clustering', model.labels_, y_train + y_test)
        print('-' * 100)