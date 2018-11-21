# Lecture 2: Image classification

## 1. KNN

Image classification : assign a label to an image 

Simplest method : calculate distance between test image and train image.

* Distance metric : L1 distance 


\begin{equation}
	d_i(I_x, I_x) = \sum\limits_p |I_1^p - I_2^p|
\end{equation}

	
Evaluate the difference between each pixel and sum the errors. 

#### Python code 

```python
import numpy as np
 
def train(X, y):
	self.X = X
    self.y = y

def predict(X):
	# Matrix of test images. Each row i s a test image. 
	num_test = X.shape(0)
    y_pred = np.zeros(num_test)
    
    for i in range(num_test):
    	distances = np.sum(np.abs(self.X, X[i:,]), axis=1)
        min_index = np.argmin(distances)
        y_pred[i] = self.y[min_index]
    
```


* Question: With N examples how fast is the training and testing 
	* Train : O(1) 
	* Test : O(N)
	* This is bad because we want fast test time and don't mind about training time

 Distance metric : L2 distance (Euclidan distance)

	$$$
	d_i(I_x, I_x) = \sqrt{\sum\limits_p (I_1^p - I_2^p)^2}
	$$$

L1 depends on coordinate system. (Linear diamond around origin)
Changes if coordinate system is rotated. 
Best if coordinates are important. Ex: bank records for people/employee record

L2 (circle around origin) : Does not depend on coordinate system. 

### 1.1 KNN hyperparameters

K and distance metric 

## 2 Choosing hyperparameters 

* Important to sperate dataset into 3 different sets 
![](http://i.markdownnotes.com/train_test_val_set.jpg)

* When the dataset is too small we can use cross validation : were the training set is split and uses to test hyperparameters 

![](http://i.markdownnotes.com/cross_validation.jpg)

## 3 Linear Classifier 

Example with CIFAR-10 dataset 
	* 10 classes
	* 50 000 training images
	* 10 000 test images 
	* image shape = 3x32x32

Linear classifier equation

$$$
	f(x, W) = Wx + b
$$$

where:
	* f(x, W) is (1, 10)
	* x is (1,3072) -> 3x32x32 stretched into 1D vector 
	* W is a (3072,10) matrix 
	* b is (1,10)

Once trained the each row of the matrix W can be reshaped and visualized to see what the network has learned. 

* A linear classifier draws linear boundaries in the dataset's dimmensional space. 
* This is a problem for datasets in which classes cannot be linearly seperable. 
