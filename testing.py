#Libaries Used
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras import activations
from tensorflow.keras import utils
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
import numpy as np
import keras
from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt

K.set_image_data_format('channels_last')

def squash(x, axis=-1):
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
    scale = K.sqrt(s_squared_norm) / (1 + s_squared_norm)
    return scale * x

def softmax(x, axis=-1):
    ex = K.exp(x - K.max(x, axis=axis, keepdims=True))
    return ex / K.sum(ex, axis=axis, keepdims=True)

def margin_loss(y_true, y_pred):
    lamb, margin = 0.5, 0.1
    return K.sum((y_true * K.square(K.relu(1 - margin - y_pred)) + lamb * (1 - y_true) * K.square(K.relu(y_pred - margin))), axis=-1)

class Capsule(Layer):
    def __init__(self,num_capsule,dim_capsule,routings=3,share_weights=True,activation='squash',**kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.share_weights = share_weights
        if activation == 'squash':
            self.activation = squash
        else:
            self.activation = activations.get(activation) 
    def get_config(self):
        config = super().get_config().copy()
        config.update({
        'num_capsule':  self.num_capsule,
        'dim_capsule' : self.dim_capsule,
        'routings':  self.routings,
        'share_weight':self.share_weights,  
        })
        return config
    def build(self, input_shape):
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.kernel = self.add_weight(
                name='capsule_kernel',
                shape=(1, input_dim_capsule,self.num_capsule * self.dim_capsule),
                initializer='glorot_uniform',
                trainable=True
                )
        else:
            input_num_capsule = input_shape[-2]
            self.kernel = self.add_weight(
                name='capsule_kernel',
                shape=(input_num_capsule, input_dim_capsule,self.num_capsule * self.dim_capsule),
                initializer='glorot_uniform',
                trainable=True)
    def call(self, inputs):
        if self.share_weights:
            hat_inputs = K.conv1d(inputs, self.kernel)
        else:
            hat_inputs = K.local_conv1d(inputs, self.kernel, [1], [1])
        batch_size = K.shape(inputs)[0]
        input_num_capsule = K.shape(inputs)[1]
        hat_inputs = K.reshape(hat_inputs,(batch_size, input_num_capsule,self.num_capsule, self.dim_capsule))
        hat_inputs = K.permute_dimensions(hat_inputs, (0, 2, 1, 3))
        b = K.zeros_like(hat_inputs[:, :, :, 0])
        for i in range(self.routings):
            c = softmax(b, 1)
            o = self.activation(keras.backend.batch_dot(c, hat_inputs, [2, 2]))
            if i < self.routings - 1:
                b = keras.backend.batch_dot(o, hat_inputs, [2, 3])
                if K.backend() == 'theano':
                    o = K.sum(o, axis=1)
        return o
    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)

batch_size = 30  
num_classes = 2
epochs = 75     

x_test=  np.load("x_test.npy")
y_test=  np.load("y_test.npy")

#model
input_image = Input(shape=(128, 128, 3))
x = Conv2D(128, (3, 3), activation='relu')(input_image)
x=BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = Capsule(64, 8, 3, True)(x)
x = Conv2D(32, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu')(x)
x = Reshape((-1, 128))(x)     
capsule = Capsule(5, 16, 3, True)(x)
output = Lambda(lambda z: K.sqrt(K.sum(K.square(z), 2)))(capsule)

model = Model(inputs=[input_image], outputs=[output])

predict=model.predict([x_test])
predict=np.argmax(predict,axis=1)

summation=0
for i in range(len(x_test)):
    if predict[i]==y_test[i]:
        summation=summation+1
        
accuracy=summation/len(x_test)

summation1=0
summation2=0

for i in range(len(x_test)):
    if predict[i]==y_test[i] and y_test[i]==0:
        summation1=summation1+1
        
specificity=summation1/np.count_nonzero(y_test==0)

for i in range(len(x_test)):
    if predict[i]==y_test[i] and y_test[i]==1:
        summation2=summation2+1
        
sensitivity=summation2/np.count_nonzero(y_test==1)

y_test = utils.to_categorical(y_test, num_classes)
y_score=model.predict([x_test])

fpr = dict()
tpr = dict()
roc = dict()
for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc[i] = auc(fpr[i], tpr[i])

fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc["micro"] = auc(fpr["micro"], tpr["micro"])

plt.rcParams.update({'font.size': 10})

plt.figure()
lw = 3
plt.plot(fpr[1], tpr[1], color='red',lw=lw, label='ROC curve (area = %0.2f)' % roc[1])
plt.plot([0, 1], [0, 1], color='blue', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()