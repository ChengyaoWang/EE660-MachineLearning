# Official Library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os, random, shutil


#Utility functions
class util():
    def __init__(self):
        print('utils class instantiated')
        pass
    
    def testReportGen(self, modelName, pred, true):
        if len(pred) != len(true):
            raise Exception('The dimensions of two inputs dont match, please check')
        acc = accuracy_score(pred, true)
        precision = precision_score(pred, true)
        recall = recall_score(pred, true)
        f1 = f1_score(pred, true)
        confusionMatrix = confusion_matrix(pred, true)
        reportFrame = 'Test Result of {0}:\nAccuracy:  {1:.4f}\nPrecision: {2:.4f}\nRecall:    {3:.4f}\n'
        reportFrame += 'F1 Score:  {4:.4f}\nConfusion Matrix:\n{5}'
        print(reportFrame.format(modelName,
                                 acc,
                                 precision,
                                 recall,
                                 f1,
                                 confusionMatrix))

    #Return [X_train, X_test, y_train, y_test, (labels)]
    def input_dataset(self, datasetName, returnType = 'numpy'):
        if not os.path.isfile('./' + datasetName + '_X_train.csv'):
            raise Exception('No Such Dataset Found, Please Check')
        if returnType == 'numpy':
            temp = [pd.read_csv('./' + datasetName + '_X_train.csv').to_numpy(),
                    pd.read_csv('./' + datasetName + '_X_test.csv').to_numpy(),
                    pd.read_csv('./' + datasetName + '_y_train.csv').to_numpy(),
                    pd.read_csv('./' + datasetName + '_y_test.csv').to_numpy(),
                    pd.read_csv('./' + datasetName + '_X_train.csv').columns.tolist()]
        elif returnType == 'pdFrame':
            temp = [pd.read_csv('./' + datasetName + '_X_train.csv'),
                    pd.read_csv('./' + datasetName + '_X_test.csv'),
                    pd.read_csv('./' + datasetName + '_y_train.csv'),
                    pd.read_csv('./' + datasetName + '_y_test.csv')]
        elif returnType == 'list':
            temp = [pd.read_csv('./' + datasetName + '_X_train.csv').values.tolist(),
                    pd.read_csv('./' + datasetName + '_X_test.csv').values.tolist(),
                    pd.read_csv('./' + datasetName + '_y_train.csv').to_numpy().ravel().tolist(),
                    pd.read_csv('./' + datasetName + '_y_test.csv').values.tolist()]
        return temp

    #Return [[X_train_x, X_test_x, y_train_x, y_test_x] * num_fold]
    def input_CVdataset(self, dirName):
        path = './' + dirName
        if not os.listdir(dirName):
            raise Exception('The Target Dir is Empty, Please Check')
        X = []
        for filename in os.listdir(path):
            if filename.startswith('fold'):
                X.append(pd.read_csv(path + '/' + filename).values.tolist())
        CV_dataset = []
        for valid_set in X:
            test_set_temp = []
            for test_set_pnt in X:
                if test_set_pnt == valid_set:
                    continue
                test_set_temp += test_set_pnt
            CV_dataset.append([test_set_temp, valid_set])
        #Seperate data and labels
        CV_dataset_zip = []
        for test_valid_tuple in CV_dataset:
            X_train = [rowPnt[:-1] for rowPnt in test_valid_tuple[0]]
            X_valid = [rowPnt[:-1] for rowPnt in test_valid_tuple[1]]
            y_train = [rowPnt[-1] for rowPnt in test_valid_tuple[0]]
            y_valid = [rowPnt[-1] for rowPnt in test_valid_tuple[1]]
            CV_dataset_zip.append([X_train, X_valid, y_train, y_valid])
        return CV_dataset_zip

    #Return [X_unlabeled, X_labeled, y_unlabeled, y_labeled], Starting Point for SSSVM
    def input_SSSVMdataset(self, ratio):
        X_train, X_valid, y_train, y_valid = self.input_CVdataset('Vani_CV_5')[0]
        fullDataset, fullLabel = X_train + X_valid, y_train + y_valid
        X_unlabeled, X_labeled, y_unlabeled, y_labeled = train_test_split(fullDataset,
                                                                          fullLabel,
                                                                          shuffle = True,
                                                                          test_size = ratio)
        zip_dataset = [X_unlabeled, X_labeled, y_unlabeled, y_labeled]
        return zip_dataset

    def scatterBoxPlot(self, pdFrame, filename, plot = False):
        g = sns.PairGrid(pdFrame, hue = 'labels', vars = pdFrame.columns.tolist()[:-1])
        g = g.map_diag(plt.hist, histtype = "step", linewidth = 1)
        g = g.map_offdiag(plt.scatter, s = 5)
        g.savefig("../Results/" + filename, dpi = 400)
        if plot:            plt.show()
    
    def coarseSelectionPlot(self, pdFrame, filename, plot = False):
        for iter in range(30):
            plt.subplot(5, 6, iter + 1)
            plt.hist([pdFrame[0][:, iter], pdFrame[1][:, iter]], color = ['red', 'blue'], histtype = 'step')
            plt.xticks([])
            plt.yticks([])
            plt.title('feature' + str(iter + 1))
        plt.subplots_adjust(wspace = 0.2, hspace = 0.5)
        plt.suptitle('Class Distribution Over Each Feature')
        plt.savefig("../Results/" + filename + '.png', dpi = 400)
        if plot:            plt.show()

    def pcaPlot(self, pyList, filename, plot = False):
        length = len(pyList[0])
        plt.plot([x + 1 for x in range(length)], pyList[0])
        plt.xlabel('Number of Principle Component')
        plt.xticks(np.arange(1, length + 1, 4))
        plt.ylabel('Variance Explained')
        for iter in range(1, length + 1, 4):
            plt.scatter(iter, pyList[0][iter - 1], s = 15, c = 'red')
            plt.text(iter, pyList[0][iter - 1] + 0.3, "%.3f" % (pyList[0][iter - 1]), fontsize = 5)
        plt.title('Results of PCA')
        plt.savefig("../Results/" + filename + '_var.png', dpi = 400)
        if plot:            plt.show()
        plt.plot([x + 1 for x in range(length)], pyList[1])
        plt.xlabel('Number of Principle Component')
        plt.xticks(np.arange(1, length + 1, 4))
        plt.ylabel('Cumulative Proportion of Variance Explained')
        for iter in range(1, length + 1, 4):
            plt.scatter(iter, pyList[1][iter - 1], s = 15, c = 'red')
            plt.text(iter, pyList[1][iter - 1] + 0.01, "%.3f" % (pyList[1][iter - 1]), fontsize = 5)
        plt.title('Results of PCA')
        plt.savefig("../Results/" + filename + '_ratio.png', dpi = 400)
        if plot:            plt.show()

    def matrixPlot(self, matrix, filename, plot = False):
        fig, ax = plt.subplots()
        ax.matshow(matrix, cmap=plt.cm.Blues)
        mxDimension = matrix.shape[0]
        for i in range(mxDimension):
            for j in range(mxDimension):
                ax.text(i, j, "%.2f" % (matrix[j, i]), size = 3, va='center', ha='center')
        plt.title(filename)
        plt.savefig('../Results/' + filename + '.png', dpi = 400)
        if plot:            plt.show()

    def Tree_plotting(self, resultList, xRange, xLabel, yLim, legendList, legendTitle, filename, plot = False):
        for result in resultList:
            plt.plot(xRange, result, linewidth = 1)
        plt.legend(legendList, loc = 'upper left', title = legendTitle)
        plt.title(filename)
        plt.ylim(yLim)
        plt.ylabel('Accuracy')
        plt.xlabel(xLabel)
        plt.savefig('../Results/' + filename + '.png', dpi = 400)
        if plot:            plt.show()

    def Linear_plotting(self, resultList, xRange, xLabel, yLim, legendList, legendTitle, filename, plot = False):
        best_y = 0
        for result in resultList:
            best_y = max(best_y, max(result))
            plt.plot(xRange, result, linewidth = 1)
        best_x = 1
        plt.legend(legendList, loc = 'lower left', title = legendTitle)
        plt.scatter(best_x, best_y, c = 'red', s = 15)
        plt.text(best_x - 2, best_y - 0.03, '({0}, {1:.3f})'.format(best_x, best_y), c = 'red')
        plt.title(filename)
        plt.ylim(yLim)
        plt.ylabel('Accuracy')
        plt.xlabel(xLabel)
        plt.savefig('../Results/' + filename + '.png', dpi = 400)
        if plot:            plt.show()

    def SpeCluster_plotting(self, resultList, xRange, yLim, legendList, filename, plot = False):
        cnt = 1
        for x, y in zip(xRange, resultList):
            plt.subplot(2, 2, cnt)
            plt.plot(x, y, linewidth = 1)
            best_x, best_y = x[y.index(max(y))], max(y)
            plt.scatter(best_x, best_y, c = 'red', s = 10)
            plt.text(best_x, best_y - 0.1, '({0}, {1:.3f})'.format(best_x, best_y), c = 'red')
            plt.xlabel(legendList[cnt - 1][1])
            plt.ylabel('Accuracy')
            plt.ylim(yLim)
            plt.title(legendList[cnt - 1][0] + '-' + legendList[cnt - 1][1])
            cnt += 1
        plt.suptitle(filename)
        plt.subplots_adjust(hspace = 0.5, wspace = 0.3)
        plt.savefig('../Results/' + filename + '.png', dpi = 400)
        if plot:            plt.show()

    def SSSVM_plotting(self, acc_record, xRange, yLim, filename, plot = False):
        iter = 1
        for x_range, ratio_result in zip(xRange, acc_record):
            plt.subplot(2, 2, iter)
            for path in ratio_result:
                plt.plot(x_range, path, linewidth = 0.5)
            plt.xlabel('Step')
            plt.ylabel('Accuracy')
            plt.ylim(yLim)
            plt.title('Labeled Data ratio = {0:.1f}'.format(0.2 * iter))
            iter += 1
        plt.suptitle(filename)
        plt.subplots_adjust(hspace = 0.5, wspace = 0.3)
        plt.savefig('../Results/' + filename + '.png', dpi = 400)
        if plot:            plt.show()

    def kMeans_plotting(self, resultList, xRange, yLim, legendList, legendTitle, filename, plot = False):
        best = []
        for result in resultList:
            plt.plot(xRange, result, linewidth = 1)
            best_x, best_y = xRange[result.index(max(result))], max(result)
            plt.scatter(best_x, best_y, c = 'red', s = 10)
            best.append([best_x, best_y])
        for best_x, best_y in best:
            plt.text(best_x, best_y - 0.1, '({0}, {1:.3f})'.format(best_x, best_y), c = 'red')
        plt.title(filename)
        plt.xlabel('LogRegStr')
        plt.ylabel('Accuracy')
        plt.legend(legendList, loc = 'lower left', title = legendTitle)
        plt.savefig('../Results/' + filename + '.png', dpi = 400)
        if plot:            plt.show()
 