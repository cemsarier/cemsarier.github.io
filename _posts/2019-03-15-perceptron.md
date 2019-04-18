---
title: "Machine Learning Project: Logictic Regression"
date: "2019-03-15"
tages: [machine learning, neural network, data science]
header:
  image: "/images/perceptron/lr.png"
excerpt: "Machine Learning, Perceptron, Data Science"
---

# Logistic Regression with a Neural Network mindset


## 1 - Packages
Lets first import our packages.
'''python
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset

%matplotlib inline
'''
## Overwiev

- a training set of m_train images labeled as cat (y=1) or non-cat (y=0)
- a test set of m_test images labeled as cat or non-cat
- each image is of shape (num_px, num_px, 3) where 3 is for the 3 channels (RGB). Thus, each image is square (height = num_px) and (width = num_px).

We will build a simple image-recognition algorithm that can correctly classify pictures as cat or non-cat.

An example is:

'''python
index = 25
plt.imshow(train_set_x_orig[index])
print ("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") +  "' picture.")
'''
y = [1], it's a 'cat' picture.
<img src="{{ site.url }}{{ site.baseurl }}/images/cat.png" alt="a cat image">

### H3 Heading

Here's some basic text.

And here's some *italics*

Here's some **bold** text.

A link [link](https://github.com)

Bulleted list:
* first item
* second
* third item

num list:
1. once
2. seconds
3. something

Python code block:

'''python
  import numpy as np

  def test_function(x, y):
    print("Hello world!")

'''
