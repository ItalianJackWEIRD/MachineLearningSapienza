import tensorflow as tf

class DQN(tf.keras.Model):
    def __init__(self, in_states, h1_nodes, out_actions):
        super(DQN, self).__init__()
        self.fc1 = tf.keras.layers.Dense(h1_nodes, activation='relu')
        self.out = tf.keras.layers.Dense(out_actions)
        self._built = False

    def build(self, input_shape):
        self.fc1.build(input_shape)
        self.out.build(self.fc1.compute_output_shape(input_shape))
        self._built = True
        super(DQN, self).build(input_shape)

    def call(self, x):
        x = self.fc1(x)
        return self.out(x)
