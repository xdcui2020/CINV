
from imblearn.over_sampling import SMOTEN
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, confusion_matrix, \
    accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt


def roc_curve_and_score(y_train, pred_proba):
    fpr, tpr, _ = roc_curve(y_train.ravel(), pred_proba.ravel())
    roc_auc = roc_auc_score(y_train.ravel(), pred_proba.ravel())
    return fpr, tpr, roc_auc
def load_data(path):
    '''
    :param path: data.csv path
    :return:
    '''
    data = pd.read_csv(path)
    return data

def data_processing(data):
    # In[] 对数据集的既往有饮酒史这列分类型变量用众数填补,这列数据缺失比例较大,还有其他几列分类型变量用众数填补
    aa = int(data['Drinking'].mode())
    data['Drinking'].fillna(aa, inplace=True)
    bb = int(data['Sleep less than 7 hours'].mode())
    data['Sleep less than 7 hours'].fillna(bb, inplace=True)
    cc = int(data['Anticipatory nausea and vomiting'].mode())
    data['Anticipatory nausea and vomiting'].fillna(cc, inplace=True)
    dd = int(data['Morning sickness'].mode())
    data['Morning sickness'].fillna(dd, inplace=True)
    ee = int(data['Use of non-prescribed antiemetics at home'].mode())
    data['Use of non-prescribed antiemetics at home'].fillna(ee, inplace=True)
    # In[] 删除特征后，缺失值占比大于0.1小于0.5的特征进行均值填补
    missing = data.isnull().sum(axis=0)
    missing_percent = missing / data.shape[0]
    # 缺失值比列大于10%的用均值填补
    list_add_means = list(missing_percent[missing_percent > 0.1].index)
    for column in list_add_means:
        data[column].fillna(data[column].mean(), inplace=True)

    # In[] 均值填补后，缺失值比列小于的10%采用KNN缺失值填补的方法
    Missing_train_1 = data.isnull().sum() / data.shape[0]
    Missing_train_1 = list(Missing_train_1[Missing_train_1 > 0].index)
    from sklearn.impute import KNNImputer

    imputer = KNNImputer(n_neighbors=3)
    df_filled = imputer.fit_transform(data[Missing_train_1])
    df_filled = pd.DataFrame(df_filled, columns=Missing_train_1)
    for i in Missing_train_1:
        data[i] = df_filled[i]

    from sklearn.preprocessing import OrdinalEncoder  # 从sklearn中导入OrdinalEncoder编码包

    encoder = OrdinalEncoder().fit(data.loc[:, ("Gender", "HED", "Antiemetic regimen")])  # 取出这些列的数据训练编码
    data.loc[:, ("Gender", "HED", "Antiemetic regimen")] = encoder.transform(
        data.loc[:, ("Gender", "HED", "Antiemetic regimen")])
    return  data

def deep_forest_model(X_resampled_smote, y_resampled_smote, test_X, test_Y):
    from deepforest import CascadeForestClassifier

    model = CascadeForestClassifier(max_layers=5, criterion='gini', n_estimators=2, n_trees=50, max_depth=None, min_samples_split=2, min_samples_leaf=1, use_predictor=False, predictor='forest', delta=1e-05)
    # model = CascadeForestClassifier(random_state=1)
    X = np.array(X_resampled_smote)
    Y = np.array(y_resampled_smote)
    model.fit(X, Y)
    gc_test_X = np.array(test_X)
    gc_yuce = model.predict(gc_test_X)
    precision = precision_score(test_Y, gc_yuce, average='weighted', )
    recall = recall_score(test_Y, gc_yuce, average='weighted')
    f1score = f1_score(test_Y, gc_yuce, average='weighted')
    accuracy = accuracy_score(test_Y, gc_yuce)
    print("gc_Accuracy:", accuracy)
    print("gc_Precision:", precision)
    print("gc_Recall:", recall)
    print("gc_f1_score：", f1score)
    return model

def rf_model(X_resampled_smote, y_resampled_smote, test_X, test_Y):
    np.random.seed(123)
    pd.options.mode.chained_assignment = None
    rf = RandomForestClassifier(max_depth=5, random_state=0)
    rf.fit(X_resampled_smote, y_resampled_smote)
    # 全部属性
    feature_names = [i for i in X_resampled_smote.columns]
    # print(feature_names)
    # 模型预测
    yuce = rf.predict(test_X)
    precision = precision_score(test_Y, yuce, average='weighted', )
    recall = recall_score(test_Y, yuce, average='weighted')
    f1score = f1_score(test_Y, yuce, average='weighted')
    accuracy = accuracy_score(test_Y, yuce)
    print("rf_Accuracy:", accuracy)
    print("rf_Precision:", precision)
    print("rf_Recall:", recall)
    print("rf_f1_score：", f1score)

    return rf

def svm_model(X_resampled_smote, y_resampled_smote, test_X, test_Y):
    from sklearn import svm
    svm_cls = svm.SVC(probability=True)
    svm_cls.fit(X_resampled_smote, y_resampled_smote)
    # 全部属性
    feature_names = [i for i in X_resampled_smote.columns]
    # print(feature_names)
    # 模型预测
    yuce = svm_cls.predict(test_X)
    precision = precision_score(test_Y, yuce, average='weighted', )
    recall = recall_score(test_Y, yuce, average='weighted')
    f1score = f1_score(test_Y, yuce, average='weighted')
    accuracy = accuracy_score(test_Y, yuce)
    print("svm_Accuracy:", accuracy)
    print("svm_Precision:", precision)
    print("svm_Recall:", recall)
    print("svm_f1_score：", f1score)
    return svm_cls
def svm_RBF_model(X_resampled_smote, y_resampled_smote, test_X, test_Y):
    from sklearn import svm
    svm_RBF_cls = svm.SVC(kernel="rbf",probability=True)
    svm_RBF_cls.fit(X_resampled_smote, y_resampled_smote)
    # 全部属性
    feature_names = [i for i in X_resampled_smote.columns]
    # print(feature_names)
    # 模型预测
    yuce = svm_RBF_cls.predict(test_X)
    precision = precision_score(test_Y, yuce, average='weighted', )
    recall = recall_score(test_Y, yuce, average='weighted')
    f1score = f1_score(test_Y, yuce, average='weighted')
    accuracy = accuracy_score(test_Y, yuce)
    print("svmRBF_Accuracy:", accuracy)
    print("svmRBF_Precision:", precision)
    print("svmRBF_Recall:", recall)
    print("svmRBF_f1_score：", f1score)

    return svm_RBF_cls
def knn_model(X_resampled_smote, y_resampled_smote, test_X, test_Y):
    from sklearn.neighbors import KNeighborsClassifier
    cls = KNeighborsClassifier(n_neighbors=2)
    cls.fit(X_resampled_smote, y_resampled_smote)
    # 全部属性
    feature_names = [i for i in X_resampled_smote.columns]
    # print(feature_names)
    # 模型预测
    yuce = cls.predict(test_X)
    precision = precision_score(test_Y, yuce, average='weighted', )
    recall = recall_score(test_Y, yuce, average='weighted')
    f1score = f1_score(test_Y, yuce, average='weighted')
    accuracy = accuracy_score(test_Y, yuce)
    print("knn_Accuracy:", accuracy)
    print("knn_Precision:", precision)
    print("knn_Recall:", recall)
    print("knn_f1_score：", f1score)

def logistic_model(X_resampled_smote, y_resampled_smote, test_X, test_Y):
    from sklearn.linear_model import LogisticRegression

    LR = LogisticRegression(penalty='l2', random_state=0)
    LR.fit(X_resampled_smote, y_resampled_smote)
    # 对测试集进行预测，pred为预测结果
    yuce = LR.predict(test_X)
    from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, \
        confusion_matrix, \
        accuracy_score

    precision = precision_score(test_Y, yuce, average='weighted')
    recall = recall_score(test_Y, yuce, average='weighted')
    f1score = f1_score(test_Y, yuce, average='weighted')
    accuracy = accuracy_score(test_Y, yuce)
    print("log_Accuracy:", accuracy)
    print("log_Precision:", precision)
    print("log_Recall:", recall)
    print("log_f1_score：", f1score)
    return LR
def naive_bayes_model(X_resampled_smote, y_resampled_smote, test_X, test_Y):
    from sklearn.naive_bayes import GaussianNB

    NB = GaussianNB()
    NB.fit(X_resampled_smote, y_resampled_smote)
    # 对测试集进行预测，pred为预测结果
    yuce = NB.predict(test_X)
    from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, \
        confusion_matrix, \
        accuracy_score

    precision = precision_score(test_Y, yuce, average='weighted')
    recall = recall_score(test_Y, yuce, average='weighted')
    f1score = f1_score(test_Y, yuce, average='weighted')
    accuracy = accuracy_score(test_Y, yuce)
    print("naive_Accuracy:", accuracy)
    print("naive_Precision:", precision)
    print("naive_Recall:", recall)
    print("naive_f1_score：", f1score)
    return NB
def xbgoost_model(X_resampled_smote, y_resampled_smote, test_X, test_Y):
    import xgboost as xgb
    model = xgb.XGBClassifier(learning_rate=0.1, n_estimators=200, max_depth=8, silent=False, random_state=0)
    model.fit(X_resampled_smote, y_resampled_smote)
    # 对测试集进行预测，pred为预测结果
    yuce = model.predict(test_X)
    from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, \
        confusion_matrix, accuracy_score
    precision = precision_score(test_Y, yuce, average='weighted', )
    recall = recall_score(test_Y, yuce, average='weighted')
    f1score = f1_score(test_Y, yuce, average='weighted')
    accuracy = accuracy_score(test_Y, yuce)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("f1_score：", f1score)
    return model

if __name__ == "__main__":
    data  = load_data("data/data.csv")
    data = data_processing(data)
    target = data['CINV']
    feature_data = data.loc[:, 'Gender':'ALP']
    train_X, test_X, train_Y, test_Y = train_test_split(feature_data, target, test_size=0.3, stratify=target)
    #X_resampled_smote, y_resampled_smote = SMOTEN(random_state=0, ).fit_resample(train_X, train_Y)
    X_resampled_smote, y_resampled_smote = train_X,train_Y
    '''
    out = []
    out = pd.DataFrame(out, columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F_1'])
    '''
    deep_forest_model_cls = deep_forest_model(X_resampled_smote, y_resampled_smote, test_X, test_Y)
    rf_model_cls = rf_model(X_resampled_smote, y_resampled_smote, test_X, test_Y)
    svm_model_cls = svm_model(X_resampled_smote, y_resampled_smote, test_X, test_Y)
    svm_RBF_model_cls = svm_RBF_model(X_resampled_smote, y_resampled_smote, test_X, test_Y)
    #knn_model(X_resampled_smote, y_resampled_smote, test_X, test_Y)
    log_mod_cls = logistic_model(X_resampled_smote, y_resampled_smote, test_X, test_Y)
   # xgboost = xbgoost_model(X_resampled_smote, y_resampled_smote, test_X, test_Y)
    NB = naive_bayes_model(X_resampled_smote, y_resampled_smote, test_X, test_Y)





    plt.figure(figsize=(8, 6))
    plt.rcParams.update({'font.size': 14})
    plt.grid()


    fpr, tpr, roc_auc = roc_curve_and_score(test_Y, deep_forest_model_cls.predict_proba(test_X)[:, 1])
    plt.plot(fpr, tpr, color='r', lw=2,
             label='AUC Deep Forest={0:.4f}'.format(roc_auc))

    fpr, tpr, roc_auc = roc_curve_and_score(test_Y, rf_model_cls.predict_proba(test_X)[:, 1])
    plt.plot(fpr, tpr, color='b', lw=2,
             label='AUC Random Forest={0:.4f}'.format(roc_auc))

    fpr, tpr, roc_auc = roc_curve_and_score(test_Y, log_mod_cls.predict_proba(test_X)[:, 1])
    plt.plot(fpr, tpr, color='lime', lw=2,
             label='AUC Logistic Regression={0:.4f}'.format(roc_auc))


    fpr, tpr, roc_auc = roc_curve_and_score(test_Y, NB.predict_proba(test_X)[:, 1])
    plt.plot(fpr, tpr, color='y', lw=2,
             label='AUC Naive Bayesmodel={0:.4f}'.format(roc_auc))

    fpr, tpr, roc_auc = roc_curve_and_score(test_Y, svm_model_cls.predict_proba(test_X)[:, 1])
    plt.plot(fpr, tpr, color='g', lw=2,
             label='AUC SVM(Linear)={0:.4f}'.format(roc_auc))

    fpr, tpr, roc_auc = roc_curve_and_score(test_Y, svm_RBF_model_cls.predict_proba(test_X)[:, 1])
    plt.plot(fpr, tpr, color='g', lw=2,
             label='AUC SVM(RBF)={0:.4f}'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.legend(loc="lower right")
    plt.title('ROC')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('1 - Specificity')
    plt.ylabel('Sensitivity')
    plt.show()

    print("END")