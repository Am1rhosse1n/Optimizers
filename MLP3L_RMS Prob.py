import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import time


def sigmoid(x):
    return  1 /( 1 + (math.e)**(-1 * x))

def sigmoid_deriviate(x):
    a = sigmoid(x)
    a = np.reshape(a,(-1,1))
    b = 1 - sigmoid(x)
    b = np.reshape(b,(-1,1))
    b = np.transpose(b)
    return np.diag(np.diag(np.matmul(a,b)))


split_ratio = 0.7
eta = 0.3
epochs = 200
epsilon = 0.1
rms1 = 0
rms2 = 0
rms3 = 0
g1 = epsilon
g2 = epsilon
g3 = epsilon
delta_w1 = 0
delta_w2 = 0
delta_w3 = 0

data = pd.read_excel('data.xlsx',header=None)
data = np.array(data)
data = data[:3726,:]


minn = np.min(data[:,0])
maxx = np.max(data[:,0])

for i in range(np.shape(data)[0]):
    for j in range(np.shape(data)[1]):
        data[i,j] = (data[i,j] - minn) / (maxx - minn)

split_line_number = int(np.shape(data)[0] * split_ratio)
x_train = data[:split_line_number,:15]
x_test = data[split_line_number:,:15]
y_train = data[:split_line_number,15]
y_test = data[split_line_number:,15]


input_dimension = np.shape(x_train)[1]
mlp_l1_neurons = 10
mlp_l2_neurons = 5
mlp_l3_neurons = 1



w1 = np.random.uniform(low=-1,high=1,size=(mlp_l1_neurons, input_dimension))
w2 = np.random.uniform(low=-1,high=1,size=(mlp_l2_neurons, mlp_l1_neurons))
w3 = np.random.uniform(low=-1,high=1,size=(mlp_l3_neurons, mlp_l2_neurons))


MSE_train = []
MSE_test = []

for i in range(epochs):


    sqr_err_epoch_train = []
    sqr_err_epoch_test = []

    output_train = []
    output_test = []


    for j in range(np.shape(x_train)[0]):
        # Feed-Forward

            # Layer 1

        net1 = np.matmul(w1,x_train[j])
        o1 = sigmoid(net1)
        o1 = np.reshape(o1,(-1,1))

            # Layer 2

        net2 = np.matmul(w2,o1)
        o2 = sigmoid(net2)
        o2 = np.reshape(o2,(-1,1))


            # Layer 3
        net3 = np.matmul(w3,o2)
        o3 = net3


        output_train.append(o3[0])

        # Error
        err = y_train[j] - o3[0]
        sqr_err_epoch_train.append(err**2)
        
        
        #Back Propagation

        
        #Gradient Descent
        f_driviate_net1 = sigmoid_deriviate(net1)
        f_driviate_net2 = sigmoid_deriviate(net2)

        delta_w1 = np.matmul(np.matmul(np.matmul(w3,f_driviate_net2),w2),f_driviate_net1)     
        delta_w1 = np.transpose(delta_w1)
        temp = np.transpose(np.reshape(x_train[j],(-1,1)))
        delta_w1 = err * -1 * 1 * np.matmul(delta_w1,temp)

        delta_w2 = np.matmul(w3,f_driviate_net2)
        delta_w2 = err * -1 * 1 * np.matmul(np.transpose(delta_w2),np.transpose(o1))
        
        delta_w3 = err * -1 * 1 * np.transpose(o2)

        w1 = np.subtract(w1 , eta/g1 * delta_w1)
        w2 = np.subtract(w2 , eta/g2 * delta_w2)
        w3 = np.subtract(w3 , eta/g3 * delta_w3)

        rms1 = rms1 + delta_w1 ** 2 
        g1 = math.sqrt(sum(sum(rms1)))
        rms2 = rms2 + delta_w2 ** 2 
        g2 = math.sqrt(sum(sum(rms2)))
        rms3 = rms3 + delta_w3 ** 2 
        g3 = math.sqrt(sum(sum(rms3)))
        

    mse_epoch_train = 0.5 * ((sum(sqr_err_epoch_train))/np.shape(x_train)[0])
    MSE_train.append(mse_epoch_train)

    for j in range(np.shape(x_test)[0]):
        # Feed-Forward

            # Layer 1

        net1 = np.matmul(w1,x_test[j])
        o1 = sigmoid(net1)
        o1 = np.reshape(o1,(-1,1))

            # Layer 2

        net2 = np.matmul(w2,o1)
        o2 = sigmoid(net2)
        o2 = np.reshape(o2,(-1,1))


            # Layer 3
        net3 = np.matmul(w3,o2)
        o3 = net3

        output_test.append(o3[0])

        # Error
        err = y_test[j] - o3[0]
        sqr_err_epoch_test.append(err ** 2)

    mse_epoch_test = 0.5 * ((sum(sqr_err_epoch_test))/np.shape(x_test)[0])
    MSE_test.append(mse_epoch_test)


    # Ploy fits

        # Train
    m_train , b_train = np.polyfit(y_train,output_train,1)

        # Test

    m_test , b_test = np.polyfit(y_test, output_test, 1)

    print(m_train,b_train,m_test,b_test)

    # Plots
    fig, axs = plt.subplots(3, 2)
    axs[0, 0].plot(MSE_train,'b')
    axs[0, 0].set_title('MSE Train')
    axs[0, 1].plot(MSE_test,'r')
    axs[0, 1].set_title('Mse Test')

    axs[1, 0].plot(y_train, 'b')
    axs[1, 0].plot(output_train,'r')
    axs[1, 0].set_title('Output Train')
    axs[1, 1].plot(y_test, 'b')
    axs[1, 1].plot(output_test,'r')
    axs[1, 1].set_title('Output Test')

    axs[2, 0].plot(y_train, output_train, 'b*')
    axs[2, 0].plot(y_train, m_train*y_train+b_train,'r')
    axs[2, 0].set_title('Regression Train')
    axs[2, 1].plot(y_test, output_test, 'b*')
    axs[2, 1].plot(y_test,m_test*y_test+b_test,'r')
    axs[2, 1].set_title('Regression Test')
    if i == (epochs - 1):
        plt.savefig('Results.jpg')
    plt.show()
    time.sleep(1)
    plt.close(fig)
