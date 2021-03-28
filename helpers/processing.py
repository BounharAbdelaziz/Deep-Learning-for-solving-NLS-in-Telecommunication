from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle

import pandas as pd
import numpy as np
from tqdm import tqdm


#----------------------------------------------------------------------------#

def fromComplexToReal(vect_im):

# vect_real = []
    N = vect_im.shape[0]
    M = vect_im.shape[1]*2
    vect_real = np.zeros((N,M))

    print("\n [INFO] {fromComplexToReal} runing...")
    print("\n [INFO] vect_real.shape : " , vect_real.shape)

    for i in tqdm(range(len(vect_im))) :
        tmp = []
        for j in range(len(vect_im[i])) :
            z = vect_im[i][j]
            tmp.append(z.real)
            tmp.append(z.imag)
        vect_real[i] = tmp

    print(" [INFO] {fromComplexToReal} vect_real.shape : ",vect_real.shape)

    return vect_real

#----------------------------------------------------------------------------#

def fromRealToComplex(vect_real):

    # init variables
    N = vect_real.shape[0]
    M = vect_real.shape[1]//2
    
    vect_im = np.zeros((N,M), dtype=np.complex64)  

    print("\n [INFO] {fromRealToComplex} runing...")

    # iterate over the vector
    for i in tqdm(range(len(vect_real))) :
        tmp = []
        j = 0

        # we can't use for loop with range() we have no control over the index
        while j < len(vect_real[i]) :

            # y = a+j*b
            im = vect_real[i][j] + 1j*vect_real[i][j+1]
            # we already processed two elements 
            j = j + 2
            # append the new element to the output array
            tmp.append(im)

        # matrix i*j
        vect_im[i] = tmp
        
    print("[INFO] {fromRealToComplex} vect_im.shape : ",vect_im.shape)

    return vect_im

#----------------------------------------------------------------------------#

def prepareDataFrame(X, y, scaling) :

    X_real = fromComplexToReal(X)
    y_real = fromComplexToReal(y)

    data = np.column_stack((X_real,y_real))

    df_ = pd.DataFrame(data)

    if scaling :
        scaler = MinMaxScaler()
        df_ = scaler.fit_transform(df_)
    
    df_ = shuffle(df_)

    print("{prepareDataFrame} df_.shape : ",df_.shape)

    return df_

#----------------------------------------------------------------------------#

def prepareTrainAndTestData(df, withReshape, ts):

    N_samples = df.shape[0]
    N_features = df.shape[1]//2
    N_cols = df.shape[1]

    if type(df) != np.ndarray :
        print("[INFO] - {prepareTrainAndTestData} converting from pandas to numpy...")
        df_np = df.to_numpy()

        # The target value is the modulated signal
        y = df_np[:N_samples,0:N_features]
        
        # The input vector is the nnget vector, ie. the distorted signal
        X = df_np[:N_samples,N_features:N_cols]

    else :
        # The target value is the modulated signal
        y = df[:N_samples,0:N_features]

        # The input vector is the nnget vector, ie. the distorted signal
        X = df[:N_samples,N_features:N_cols]

    N_features = len(X[0])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ts)

    N_samples_train = len(X_train)
    N_samples_test = len(X_test)

    if withReshape :
        X_train = X_train.reshape(N_samples_train, N_features, 1)
        X_test = X_test.reshape(N_samples_test, N_features, 1)
        y_train = y_train.reshape(N_samples_train, N_features, 1)
        y_test = y_test.reshape(N_samples_test, N_features, 1)

    print("[INFO] train and test data are ready.")
    print("X.shape : ",X.shape)
    print("y.shape : ",y.shape)
    print("X_train.shape : ",X_train.shape)
    print("y_train.shape : ",y_train.shape)
    print("X_test.shape : ",X_test.shape)
    print("y_test.shape : ",y_test.shape)

    return X_train, X_test, y_train, y_test

#----------------------------------------------------------------------------#