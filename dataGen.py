# Official Library
import os, random, shutil
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA, KernelPCA
from sklearn.model_selection import StratifiedKFold
# Other Python Files
from utils import util

#Class for data Preprocessing and Generation
class dataGenerator():
    fileList = ['wdbc.csv']
    def __init__(self, CV_fold = 5, produceResult = False, forceRegen = False):
        self.forceRegen = forceRegen
        if self.forceRegen:
            print('Dataset force regeneration is active, regenerating all datasets')
        elif not self.forceRegen:
            print('Dataset force regeneration deactivated, using existing datasets')
        #Create working directories
        self.dirCreate()
        #Instantiate utility class
        self.utils = util()
        self.readData()
        self.raw_standarization()
        self.class_split()
        if produceResult or self.forceRegen:
            self.correlationMx()
            self.CV_feature_plot()
            self.feature_selection()
        self.CV_datasetGen(fold = CV_fold, pca_type = 'Vani', test_ratio = 0.2)
        self.CV_datasetGen(fold = CV_fold, pca_type = 'Ker', test_ratio = 0.2)
    
    #Create working directory
    def dirCreate(self):
        os.chdir('./Dataset')
        if not os.path.isdir('./ReadyToUseDataset'):
            os.mkdir('./ReadyToUseDataset')
        os.chdir('./ReadyToUseDataset')
        if not os.path.isdir('../Results'):
            os.mkdir('../Results')
        print('Current Working Directory:' + os.getcwd())

    #Read the original CSV file and divide the training set & labels
    def readData(self):
        dataset = pd.read_csv('../wdbc.csv')
        if not os.path.isfile('./raw_X_train.csv') or self.forceRegen:
            print('No Existing Dataset Found, Creating one......')
            #Create raw dataset, spliting the training set and labels
            cols = dataset.columns.tolist()
            raw_id = dataset[cols[0]]
            raw_label = dataset[cols[1]].to_numpy()
            raw_dataset = dataset[cols[2:-2]]
            #Encode the Labels
            label_encoder = LabelEncoder().fit(raw_label)
            if len(label_encoder.classes_) != 2: raise Exception('The labels contrains more than 2 labels, please check')
            raw_label = label_encoder.transform(raw_label)
            raw_label = pd.DataFrame(raw_label, columns = ['label'])
            #Split into train / test dataset, roughly 4 : 1
            #We want to keep the prior probability roughly the same in train / test
            ratio_train, ratio_test = 0, 1
            while abs(ratio_train - ratio_test) > 0.02:
                mask = np.random.rand(len(raw_dataset)) < 0.8
                y_train, y_test = raw_label[mask].to_numpy(), raw_label[~mask].to_numpy()
                ratio_train, ratio_test = float(sum(y_train)) / len(y_train), float(sum(y_test)) / len(y_test)
            X_train, X_test = raw_dataset[mask], raw_dataset[~mask]
            print('Dataset Split Completed')
            #Write to CSV
            X_train.to_csv('./raw_X_train.csv', index = False, header = True)
            X_test.to_csv('./raw_X_test.csv', index = False, header = True)
            pd.DataFrame(y_train, columns = ['labels']).to_csv('./raw_y_train.csv', index = False, header = True)
            pd.DataFrame(y_test, columns = ['labels']).to_csv('./raw_y_test.csv', index = False, header = True)
            self.dataset_size = [X_train.shape[0], X_test.shape[0]]
            self.feature_num = X_train.shape[1]
            print('Raw Train / Test Dataset Created')
            print('Training Set Size:', self.dataset_size[0], '\n'
                  'Testing Set Size:', self.dataset_size[1], '\n'
                  'Class Ratio in Training: {0:.3f}'.format(ratio_train), '\n'
                  'Class Ratio in Testing: {0:.3f}'.format(ratio_test))
        elif os.path.isfile('./raw_X_train.csv'):
            print('Found Exsiting Dataset')
            y_train, y_test = pd.read_csv('./raw_y_train.csv').to_numpy(), pd.read_csv('./raw_y_test.csv').to_numpy()
            print('Training Set Size:', y_train.shape[0], '\n'
                  'Testing Set Size:', y_test.shape[0], '\n'
                  'Class Ratio in Training: {0:.3f}'.format(float(sum(y_train)) / len(y_train)), '\n'
                  'Class Ratio in Testing: {0:.3f}'.format(float(sum(y_test)) / len(y_test)))
    
    #Produce a dataset after standardization (mean std, not min-max std)
    def raw_standarization(self):
        X_train, X_test, y_train, y_test = self.utils.input_dataset('raw', 'pdFrame')
        #Save as class attribute
        self.mean, self.std = X_train.mean(), X_train.std()
        self.feature_num = len(X_train.columns.tolist())
        std_X_train = (X_train - X_train.mean()) / X_train.std()
        std_X_test = (X_test - X_train.mean()) / X_train.std()
        std_X_train.to_csv('./std_X_train.csv', index = False, header = True)
        std_X_test.to_csv('./std_X_test.csv', index = False, header = True)
        shutil.copyfile('./raw_y_test.csv', './std_y_test.csv')
        shutil.copyfile('./raw_y_train.csv', './std_y_train.csv')
        #Store it as a class variable
        print('Standardized Dataset Created')

    #Create class disjoint datasets
    def class_split(self):
        X, _, y, _ = self.utils.input_dataset('std', 'pdFrame')
        X_label, y_label = X.columns.tolist(), y.columns.tolist()
        X, y = X.values.tolist(), y.values.tolist()
        pos_data, neg_data = [], []
        for iter in range(len(X)):
            if y[iter][0] == 1:
                pos_data.append(X[iter])
            else:
                neg_data.append(X[iter])
        pd.DataFrame(pos_data, columns = X_label).to_csv('./split_pos.csv', index = False, header = True)
        pd.DataFrame(neg_data, columns = X_label).to_csv('./split_neg.csv', index = False, header = True)
        print('Class Disjoint Dataset Created')

    #Calculate and plot the correlation Matrix of the feastures
    def correlationMx(self):
        X, _, _, _, _ = self.utils.input_dataset('raw', 'numpy')
        corMx = np.matmul(X.transpose(), X) / X.shape[0]
        self.utils.matrixPlot(corMx, 'Correlation Matrix Plot before STD', False)
        X_std, _, _, _, _ = self.utils.input_dataset('std', 'numpy')
        corMx_std = np.matmul(X_std.transpose(), X_std) / X_std.shape[0]
        self.utils.matrixPlot(corMx_std, 'Correlation Matrix Plot after STD', False)
        print('Correlation Matrix Heat Map created at ./Results')

    #Calculate the Coefficient of Variation of Each Feature
    def CV_feature_plot(self):
        #Calculate the Coefficient of Variation of Each Feature
        coef_var_List = []
        for iter in range(self.feature_num):
            coef_var_List.append([self.std[iter] / self.mean[iter], iter])
        coef_var_List.sort(key = lambda x: x[0], reverse = True)
        num_selection = int(np.sqrt(self.feature_num)) # 5 in this case
        #Get the index of the selected features, and generate a new dataset
        selected_index = ["'" + str(coef_var_List[iter][1]) + "'" for iter in range(num_selection)]
        X, _, y, _ = self.utils.input_dataset('raw', 'pdFrame')
        dataset = pd.concat([X[selected_index], y], axis = 1)
        print('Features selected for Coefficient Variation Plotting:', selected_index)
        #Plot the scatter matrix & box plots of the five features, before STD
        self.utils.scatterBoxPlot(dataset, 'Scatter-Box before STD.png')
        #Plot the scatter matrix & bos plots of the five features, after STD
        X, _, y, _ = self.utils.input_dataset('std', 'pdFrame')
        dataset = pd.concat([X[selected_index], y], axis = 1)
        self.utils.scatterBoxPlot(dataset, 'Scatter-Box after STD.png')
        print('Scatter-Box plot of Top-5 Coefficient Variation Feature created at ./Results')

    #Step1: Plot class seperate box plots, select 25 out of 30
    #Step2: Do PCA (Vanilla & Kernel version)
    def feature_selection(self):
        #We often use STD data to perform feature selection, especially PCA
        #To avoid variation split dominated by some features
        X_pos = pd.read_csv('./split_pos.csv').to_numpy()
        X_neg = pd.read_csv('./split_neg.csv').to_numpy()
        self.utils.coarseSelectionPlot([X_pos, X_neg], 'Coarse Feature Selection', True)
        self.eliminatedFeature = ["'10'", "'12'", "'15'", "'19'", "'20'"]
        print('Coarse Feature Selection Completed, Eliminated Features No.', self.eliminatedFeature)
        #Do Vanilla PCA
        X_train, X_test, _, _ = self.utils.input_dataset('std', 'pdFrame')
        X_train = X_train.drop(self.eliminatedFeature, axis = 1).to_numpy()
        X_test = X_test.drop(self.eliminatedFeature, axis = 1).to_numpy()
        pca_instance = PCA(n_components = X_train.shape[1]).fit(X_train)
        explained_variance = pca_instance.explained_variance_
        explained_ratio = []
        for iter in range(X_train.shape[1]):
            explained_ratio.append(sum(pca_instance.explained_variance_ratio_[:iter + 1]))
        self.utils.pcaPlot([explained_variance, explained_ratio], 'PCA result', True)
        print('PCA Result Plot created at ./Results')
        #Choose n_components = 17, which has 99.7% variance, also just suitable amount for various models
        #Output the Dataset
        self.output_dimension = 17
        column = []
        for iter in range(self.output_dimension):
            column.append('proj_fea_' + str(iter + 1))
        pca_instance = PCA(n_components = self.output_dimension).fit(X_train)
        X_train = pca_instance.transform(X_train)
        X_test = pca_instance.transform(X_test)
        pd.DataFrame(X_train, columns = column).to_csv('./VaniPCA_X_train.csv', index = False, header = True)
        pd.DataFrame(X_test, columns = column).to_csv('./VaniPCA_X_test.csv', index = False, header = True)
        shutil.copyfile('./raw_y_train.csv', './VaniPCA_y_train.csv')
        shutil.copyfile('./raw_y_test.csv', './VaniPCA_y_test.csv')
        print('Dataset after Vanilla PCA created')
        #Do Kernel PCA, unsupervised-method, only uses data, no label, make it consistent with Vanilla version
        X_train, X_test, _, _ = self.utils.input_dataset('std', 'pdFrame')
        X_train = X_train.drop(self.eliminatedFeature, axis = 1).to_numpy()
        X_test = X_test.drop(self.eliminatedFeature, axis = 1).to_numpy()
        pca_instance = KernelPCA(n_components = self.output_dimension, kernel = 'rbf').fit(X_train)
        X_train = pca_instance.transform(X_train)
        X_test = pca_instance.transform(X_test)
        pd.DataFrame(X_train, columns = column).to_csv('./KerPCA_X_train.csv', index = False, header = True)
        pd.DataFrame(X_test, columns = column).to_csv('./KerPCA_X_test.csv', index = False, header = True)
        shutil.copyfile('./raw_y_train.csv', './KerPCA_y_train.csv')
        shutil.copyfile('./raw_y_test.csv', './KerPCA_y_test.csv')
        print('Dataset after Kernel PCA created')
        print('Feature Selection Completed, Remaining Number of Feature:', self.output_dimension)
    
    '''
    #Deprecated, Now in utils.input_dataset() & utils,input_CVdataset
    #Fetch Data
    #Return:[X_train, X_test, y_train, y_test], all numpy arrays
    def dataFetch(self, filename, test_ratio):
        path_dataset = './' + filename[0] + '.csv'
        path_label = './' + filename[1] + '.csv'
        if not os.path.isfile(path_dataset) or not os.path.isfile(path_label):
            raise Exception('Target Dataset / Label Not Found, Please Check')
        X, y = pd.read_csv(path_dataset).to_numpy(), pd.read_csv(path_label).to_numpy().ravel()
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size = test_ratio, 
                                                            shuffle = True)
        print('Class Ratio in Train:', float(sum(y_train)) / len(y_train))
        print('Class Ratio in Test:', float(sum(y_test)) / len(y_test))
        return X_train, X_test, y_train, y_test
    '''
    #Generate / Fetch Dataset for Cross Validation
    #Drawn from dir: ./final_CV_fold
    #Return [cv1, cv2, ......, cvfold, test], all numpy arrays, labels attached in the last column
    def CV_datasetGen(self, fold = 5, pca_type = 'Vani', test_ratio = 0.2):
        #If a certain fold already exists
        path = './' + pca_type + '_CV_' + str(fold)
        if not os.path.isdir(path):
            os.mkdir(path)
        returnList = []
        if len(os.listdir(path)) == 0 or self.forceRegen:
            if not os.path.isfile('./' + pca_type + 'PCA_X_train.csv'):
                raise Exception('Dataset After Preprocessing not found, please run dataGenerator.feature_selection() first')
            X, _, y, _, labels = self.utils.input_dataset(pca_type + 'PCA', 'numpy')
            labels += ['labels']
                #y = y.ravel()
            skf_instance = StratifiedKFold(n_splits = fold, shuffle = True)
            cnt, temp = 1, []
            for _, test_index in skf_instance.split(X, y):
                for instance_pnt in test_index:
                    row = X[instance_pnt].tolist()
                    row.append(y[instance_pnt][0])
                    temp.append(row)
                pd.DataFrame(temp, columns = labels).to_csv(path + '/fold_' + str(cnt) + '.csv',
                                                            index = False,
                                                            header = True)
                cnt += 1
                temp = []
            shutil.copyfile('./' + pca_type + 'PCA_X_test.csv', path + '/X_test.csv')
            shutil.copyfile('./' + pca_type + 'PCA_y_test.csv', path + '/y_test.csv')
        for filename in os.listdir(path):
            if filename.endswith('.csv'):
                returnList.append(pd.read_csv(path + '/' + filename).to_numpy())
        print(pca_type + ' Dataset Fold = {0} Created'.format(fold))
        return returnList
