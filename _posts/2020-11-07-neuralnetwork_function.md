---
title: "A Neural Network Function v.1"
date: "2020-11-07"
tages: [algorithm, neural networks , keras, tensowflow]
excerpt: "Algorithm, Neural Netowrks, Keras, TensowFlow"
categories:
- Algorithm
- Neural Networks
- Keras
- TensorFlow
---

# About
In this function, user can enter some parameters into the given function and try different combinations easily. The purpose was to build an auto neural class for classification problems. We already have the sklearn auto modelling part ready in another class. But neural network part should be seperate since our main class' design requires it to. Auto neural network is not easy, also probably it's not going to work efficiently. However, for our case, we generally work with similar datasets and similar feature architecture. Thus, a level of success can be achieved (hopefully) in the future updates.

First, we need to create a function that builds a neural network, then we need to optimize it. This is the first attempt of the first part. The general working principle is, according to layer size and neurons entered by user, this function creates a sequential model from Keras library and adds layers one by one. First it creates the input layer according to the shape of x_train dataframe. Then it adds the other layers (if they exist) with the help of a loop.

After the compilation of the model, we create a *history* object, which stores the training information. And the function prints the training auc ve loss graphs. Note that we train with validation. So that the x_test and y_test parameters are validation datasets.


## The Code

Necessary libraries:

```python
from tensorflow.keras.metrics import AUC
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l1
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

```

* Most of the modelling task I work, **roc_auc** score is the main metric. So I train for it.
* The **regularization** part only takes one input and applies the same amount of regularization to each layer.

```python
#Neural Network NN
def fit_nn_model(x_train, y_train, x_test, y_test, layer_size = 2, units =[16,16], epochs = 10, batch_size=128, regularization = 0.02):

    print("Neural network model fitting started.\n")
    print("Preparing your model...\n")

    if batch_size > x_train.shape[0]: #just to make sure that user inputs a logical batch size
        batch_size = x_train.shape[0]
    #Warnings
    if len(units)==layer_size: #this function will only work if list of units' size matches number of layers
        #Create sequential object
        model = Sequential()
        model.add(Dense(units= units[0], input_shape = [x_train.shape[1]], activation = 'relu', activity_regularizer=l1(regularization)))
        if len(units)>1:
            for unit in range(len(units)-1):
                model.add(Dense(units= units[unit+1], activation = 'relu', activity_regularizer=l1(regularization)))
        model.add(Dense(units=1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy',optimizer='adam', metrics=[AUC()])

        #training with validation
        history = model.fit(x_train, y_train,validation_data = (x_test,y_test), epochs=epochs, batch_size = batch_size)
        pred = [1 if x>=0.5 else 0 for x in model.predict(x_test)[:,-1]]

        #check_df = pd.DataFrame({'pred':nn_model.predict(x_test)[:,-1], 'pred2':pred})       
        print("Confusion matrix: \n")
        cm = confusion_matrix(y_test,pred)
        ax= plt.subplot()
        sns.heatmap(cm, annot=True, ax = ax,cmap='YlGnBu',fmt='g')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')
        plt.show()
        print()

        print("Classification Report: \n")
        print(classification_report(y_test,pred))
        print()       

        plt.plot(history.history[list(history.history.keys())[1]])
        plt.plot(history.history[list(history.history.keys())[3]])
        plt.title('Model accuracy')
        plt.ylabel('AUC Score')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()      
        print()

        print()
        print("Train gini is: %.3f" % (2*history.history[list(history.history.keys())[1]][-1] -1))
        print("Validation gini is: %.3f" % (2*history.history[list(history.history.keys())[3]][-1] -1))
        print("\nNeural network model fitting finished.\n")
        return model

    else:
        print("WARNING: Fitting failed.")
        print("Please make sure units and layer_size matches correctly and train again")
```

For small to medium datasets with small size of feature set (<100), this function runs in less than 10 seconds. With some base level knowledge about neural networks and model training, user can set sensible parameters and train a well-performing simple model rapidly. The only problem is, due to random initialization and regularization, the outputs are not reproducible given the same parameters. We will work on that in the second part of this project.
