import numpy
import numpy as np

#function f = w * x
X = numpy.array([1,2,3,4],dtype=np.float32)
y = numpy.array([2.4,6.8],dtype=np.float32)

w = 0.0

#model prediction
def forward(x):
    return w * x

#loss
def loss(y, y_predicted):
    return ((y_predicted-y)**2).mean()

def gradiennt(x, y, y_predicted):
    return np.dot(2*x, y_predicted-y).mean()

print(f'prediction before training:  {forward(5)}')

learning_rate = 0.01
n_iter = 10

for epoch in range(n_iter):
    y_pred = forward(X)

    #loss
    l = loss(y,y_pred)

    #gradient
    dw = gradiennt(X,y,y_pred)

    #update weights
    w -= learning_rate *dw

    if epoch %1 == 0:
        print(f'epoch {epoch + 1} ')

print(f'prediction before training:  {forward(5)}')