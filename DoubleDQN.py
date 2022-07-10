import tensorflow as tf

class Qnetwork():
    def __init__(self, s_size, a_size):
        self.a_size = a_size
        self.s_size = s_size
        #self.layer_name = layer_name
        self.structure()


    def structure(self):
        init = tf.compat.v1.glorot_normal_initializer()
        tf.compat.v1.disable_eager_execution()

        # 输入层：采用当前位置的地理编码并除以180
        self.input = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None,self.s_size])
        self.inputt = tf.truediv(self.input,[[180.0,180.0]])

        # 全连接层（10个神经元）
        self.W_1 = tf.compat.v1.get_variable(shape=[self.s_size,10], dtype=tf.float32, name='w1', initializer=init)
        self.b_1 = tf.compat.v1.get_variable(shape=[10], dtype=tf.float32, name='b1')
        h_1 = tf.nn.relu(tf.matmul(self.inputt, self.W_1) + self.b_1)  # relu激活函数
        self.h_1_drop = tf.nn.dropout(h_1, keep_prob=0.75)       # 丢失率为 0.25

        # 全连接层（6个神经元）
        self.W_2 = tf.compat.v1.get_variable(shape=[10,6], dtype=tf.float32, name='w2', initializer=init)
        self.b_2 = tf.compat.v1.get_variable(shape=[6], dtype=tf.float32, name='b2')
        h_2 = tf.nn.relu(tf.matmul(h_1, self.W_2) + self.b_2)
        self.h_2_drop = tf.nn.dropout(h_2, keep_prob=0.75)

        # 最后一层 ： 四维输出代表每个动作的 Q 值
        self.W_3 = tf.compat.v1.get_variable(shape=[6, self.a_size], dtype=tf.float32, name='w3', initializer=init)
        self.action = tf.matmul(self.h_2_drop, self.W_3)   # 每个动作的Q值
        self.predict = tf.argmax(input=self.action, axis=1)

        self.target_y = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None])
        self.a = tf.compat.v1.placeholder(dtype=tf.int32, shape=[None])
        self.predict_onehot = tf.one_hot(indices=self.a, depth=4, on_value=1, off_value=0)

        self.floatpre = tf.cast(self.predict_onehot, tf.float32)

        #self.Q = tf.matmul(self.floatpre,self.action)
        self.Q = tf.reduce_sum(tf.multiply(self.floatpre,self.action), axis=1)   # Targetnet's Q value in batch
        '''
            损失是通过减去Q值计算的
            Q值是通过将当前状态输入到Q网络中并从目标值 y 中选择对应动作的值来计算的
        '''
        self.error = tf.square(self.target_y - self.Q)
        self.loss = tf.reduce_mean(self.error)
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.0001)
        self.update = self.optimizer.minimize(self.loss)

