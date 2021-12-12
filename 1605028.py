import numpy as np
import pandas as pd
import math
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
np.random.seed(9999)


def WeightedMajority(h, z, data):
    h = np.array(h)
    z = np.array(z)
    # print('z is ', z)
    # print('h shape is ', h.shape)
    data = data.transpose()
    # print(' data shape is ', data.shape)
    M1 = h@data
    # print(' m1 shape is ', M1.shape)
    p = np.tanh(M1)
    # print(' p  is ', p)
    # print(' P shape is ', p.shape)
    # print('  Z shape is ', z.shape)
    predict = z@p

    # print(' predict shape is ', predict.shape)
    predict = np.sign(predict)
    return predict


def confusionFunctions(yTrue, yPredict):
    tn, fp, fn, tp = confusion_matrix(yTrue, yPredict).ravel()
    # print(tn, " ", fp, " ", fn, " ", tp)
    accuracy = (tp + tn)/(tp+fp+fn+tn)
    # print('accuracy : ', accuracy)

    sensitivity = tp/(tp+fn)

    specificity = tn/(tn+fp)

    precision = tp/(tp+fp)

    fdr = fp/(fp+tp)

    f1_score = 2*tp/(2*tp + fp + fn)
    p = [accuracy, sensitivity, specificity, precision, fdr, f1_score]
    return p


def logisticRegression(data, result, flag):

    rowNum = data.shape[0]
    colNum = data.shape[1]
    checker = 100000000
    oneMatrix = np.ones(rowNum)
    # weight = np.random.rand(colNum)
    weight = np.zeros(colNum)
    counter = 0
    while (checker > 0.5 and counter < 50000):
        counter = counter + 1
        if(counter % 10000 == 0):
            print(counter)
        alpha = 0.01
        tempDf = data.to_numpy()
        Yp = np.tanh(tempDf.dot(weight))
        #   Y = (df.iloc[:,-1]).to_numpy()
        Y = result
        M1 = 1 - Yp ** 2
        #   print(M1)
        M2 = np.multiply(Y - Yp, M1)
        M2 = alpha * M2

        transposedDf = tempDf.transpose()
        r = transposedDf.dot(M2)
        #         weight= weight+ np.multiply(r,alpha)
        weight = weight + r
        checker = np.sum((Y-Yp)**2)/(2 * rowNum)
    # print(checker)
    #     print(weight)

    predicted = np.sign(np.tanh(tempDf.dot(weight)))
    rs = confusionFunctions(result, predicted)
    if (flag == 1):
        return rs

    #     acc= accuracy_score(Y_out, predicted)
    #     print("accuracy value " ,acc)
    #     print(weight)
    return weight


def adaboost(df, iteration, Y_out):
    rowNum = df.shape[0]
    colNum = df.shape[1]
    # print(df.iloc[0])
    w = np.empty(rowNum)
    w.fill(1/rowNum)
    h = []
    z = []
    for k in range(iteration):
        df['ans'] = Y_out

        data = df.sample(n=len(df), replace=True, weights=w, random_state=97)
        result = data['ans'].to_numpy()

        df.drop('ans', inplace=True, axis=1)
        data.drop('ans', inplace=True, axis=1)
        wL = logisticRegression(data, result, 2)
        predicted = np.sign(np.tanh((data.to_numpy()).dot(wL)))

        error = 0
        for j in range(len(data)):
            if (predicted[j] != result[j]):
                error = error + w[j]
        if (error > 0.5):
            continue
        h.append(wL)

        for j in range(len(data)):
            if(predicted[j] == result[j]):
                w[j] = w[j] * (error)/(1-error)

        # Normalize W
        s = np.sum(w)
        w = w/s
        # Finding z
        q = math.log2((1-error)/error)
#         print('q is ',q )
        z.append(q)
    predict = WeightedMajority(h, z, df)
    out = confusionFunctions(Y_out, predict)
    return out


def dataPreprocessing(df):
    # filling nan values-  --stackoverflow
    catCols = df.select_dtypes(include=['object']).columns.tolist()
    for column in df:
        if df[column].isnull().any():
            if(column in catCols):
                df[column] = df[column].fillna(df[column].mode()[0])
            else:
                df[column] = df[column].fillna(df[column].mean)
    # One hot encoding
    rowNum = df.shape[0]
    df = pd.get_dummies(df, drop_first=True)

    # standardize
    st = StandardScaler()
    df.iloc[:, :] = st.fit_transform(df.iloc[:, :])

    # add a  1 column  to dataframe beginning
    dummyCol = np.ones(rowNum)
    df.insert(loc=0, column='dummy', value=dummyCol)

    # train and test split
    train, test = train_test_split(df, test_size=0.2, random_state=99)
    df = train

    return df


def runDataset1():
    df = pd.read_csv(
        "WA_Fn-UseC_-Telco-Customer-Churn.csv", na_values=[' ', ' ?'])
    df.drop('customerID', inplace=True, axis=1)

    df = dataPreprocessing(df)
    Y_out = df.iloc[:, -1]      # i final output
    Y_out = np.sign(Y_out)  # convert to -1 and +1
    df.drop(df.columns[-1], axis=1, inplace=True)  # drop the last column

    print('________ Logistic Regression Outuput __________')
    logisticOutput = logisticRegression(df, Y_out, 1)
    # print('logistic output ', logisticOutput)
    print('accuracy : ', logisticOutput[0])
    print('sensitivity : ', logisticOutput[1])
    print('specificity:   ', logisticOutput[2])
    print('precision : ', logisticOutput[3])
    print('fdr  ', logisticOutput[4])
    print('f1 score ', logisticOutput[5])

    print('________ Adaboost Outuput __________')
    adaboostOutput = adaboost(df, 5, Y_out)
    print('accuracy : ', adaboostOutput[0])

    print('1')


def runDataset2():
    df = pd.read_csv("adult.data", na_values=[" ", ' ?', np.NaN])
    df.columns = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship", "race",
                  "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]
    df = dataPreprocessing(df)
    Y_out = df.iloc[:, -1]      # i final output
    Y_out = np.sign(Y_out)  # convert to -1 and +1
    df.drop(df.columns[-1], axis=1, inplace=True)  # drop the last column

    print('________ Logistic Regression Outuput __________')
    logisticOutput = logisticRegression(df, Y_out, 1)
    # print('logistic output ', logisticOutput)
    print('accuracy : ', logisticOutput[0])
    print('sensitivity : ', logisticOutput[1])
    print('specificity:   ', logisticOutput[2])
    print('precision : ', logisticOutput[3])
    print('fdr  ', logisticOutput[4])
    print('f1 score ', logisticOutput[5])

    print('________ Adaboost Outuput __________')
    adaboostOutput = adaboost(df, 5, Y_out)
    print('accuracy : ', adaboostOutput[0])


def runDataset3():
    df = pd.read_csv("creditcard.csv", na_values=[" ", ' ?', np.NaN])

    df = dataPreprocessing(df)
    Y_out = df.iloc[:, -1]      # i final output
    Y_out = np.sign(Y_out)  # convert to -1 and +1
    df.drop(df.columns[-1], axis=1, inplace=True)  # drop the last column

    print('________ Logistic Regression Outuput __________')
    logisticOutput = logisticRegression(df, Y_out, 1)
    # print('logistic output ', logisticOutput)
    print('accuracy : ', logisticOutput[0])
    print('sensitivity : ', logisticOutput[1])
    print('specificity:   ', logisticOutput[2])
    print('precision : ', logisticOutput[3])
    print('fdr  ', logisticOutput[4])
    print('f1 score ', logisticOutput[5])

    print('________ Adaboost Outuput __________')
    adaboostOutput = adaboost(df, 5, Y_out)
    print('accuracy : ', adaboostOutput[0])


def main():
    while 1 == 1:
        print("\n\n\n")
        a = int(input("Enter a number from 1 to 3 :  "))
        if a == 1:
            runDataset1()

        elif a == 2:
            runDataset2()

        elif a == 3:
            runDataset3()
        else:
            continue


if __name__ == "__main__":
    main()
