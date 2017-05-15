import tensorflow as tf
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as mp

# Translate a list of labels into an array of 0's and one 1.
# i.e.: 4 -> [0,0,0,0,1,0,0,0,0,0]
def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


data = np.genfromtxt('iris.data', delimiter=",")  # iris.data file loading
np.random.shuffle(data)  # we shuffle the data
#Conjunto de Entrenamiento
x_data_training = data[0:107, 0:4].astype('f4')  # the samples are the four first rows of data
y_data_training = one_hot(data[0:107, 4].astype(int), 3)  # the labels are in the last row. Then we encode them in one hot code
#Conjunto de Test
x_data_test = data[107:129, 0:4].astype('f4')  # the samples are the four first rows of data
y_data_test = one_hot(data[107:129, 4].astype(int), 3)  # the labels are in the last row. Then we encode them in one hot code
#Conjunto de Validacion
x_data_valid = data[129:151, 0:4].astype('f4')  # the samples are the four first rows of data
y_data_valid = one_hot(data[129:151, 4].astype(int), 3)  # the labels are in the last row. Then we encode them in one hot code

print "\nSome samples..."
for i in range(20):
    print x_data_training[i], " -> ", y_data_training[i]
print

x = tf.placeholder("float", [None, 4])  # samples
y_ = tf.placeholder("float", [None, 3])  # labels

W1 = tf.Variable(np.float32(np.random.rand(4, 5)) * 0.1)
b1 = tf.Variable(np.float32(np.random.rand(5)) * 0.1)

W2 = tf.Variable(np.float32(np.random.rand(5, 3)) * 0.1)
b2 = tf.Variable(np.float32(np.random.rand(3)) * 0.1)

h = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
# h = tf.matmul(x, W1) + b1  # Try this!
y = tf.nn.softmax(tf.matmul(h, W2) + b2)

loss = tf.reduce_sum(tf.square(y_ - y))

train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)  # learning rate: 0.01

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

print "----------------------"
print "   Start training...  "
print "----------------------"

batch_size = 20
valid_error= 0.1;
train_errors = []
valid_errors = []
current_valid_error = 1
epoch = 0
current_diff = 100
test_errors = []
train_error = 0.1

while(0.1 <= train_errors and current_diff > 0.001):
    epoch+=1
    for jj in xrange(len(x_data_training) / batch_size):
        batch_train_xs = x_data_training[jj * batch_size: jj * batch_size + batch_size]
        batch_train_ys = y_data_training[jj * batch_size: jj * batch_size + batch_size]
        sess.run(train, feed_dict={x: batch_train_xs, y_: batch_train_ys})

    train_error = sess.run(loss, feed_dict={x: batch_train_xs, y_: batch_train_ys})
    train_errors.append(train_error)

    if(epoch > 1):
        current_diff = train_errors[-2] - train_error

    current_valid_error = train_error


    print "Validacion------->"
    print "Epoch #:", epoch, "Error: ", current_valid_error
    result = sess.run(y, feed_dict={x: x_data_valid})
    #for b, r in zip(batch_valid_ys, result):
    #    print b, "-->", r


print "----------------------"
print "   Finish training...  "
print "----------------------"

num_fallos = 0
num_aciertos = 0



print "Test------->",
result = sess.run(y, feed_dict={x: x_data_test})
for b, r in zip(y_data_test, result):
    if (np.argmax(b) != np.argmax(r)):
        num_fallos += 1
    else:
        num_aciertos += 1
print "Aciertos: ", num_aciertos, "Fallos: ", num_fallos

mp.ylabel("Error")
mp.xlabel("Epochs")

graph_train, = mp.plot(train_errors)

mp.legend(handles=[graph_train],
labels=["Training errors"])
mp.savefig('iris.png')

