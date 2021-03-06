from keras import layers, models, optimizers, backend as K

from agents.neural import dense


class Critic:
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, learning_rate=None):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            learning_rate (float): Optimizer learning rate
        """
        self.state_size = state_size
        self.action_size = action_size

        # Initialize any other variables here
        self.learning_rate = learning_rate
        self.build_model()

    def build_model(self):
        """
        Build a critic (value) network that maps
        (state, action) pairs -> Q-values.
        """
        # Define input layers
        states = layers.Input(shape=(self.state_size,), name='states')
        actions = layers.Input(shape=(self.action_size,), name='actions')

        # Add hidden layer(s) for state pathway
        # Simplified neural net compared to the original DDPG paper.
        net_states = dense(states, 64)
        net_actions = dense(actions, 64)

        # Combine state and action pathways
        net = layers.Add()([net_states, net_actions])
        net = layers.BatchNormalization()(net)
        net = layers.LeakyReLU(alpha=0.1)(net)

        # Simplified neural net compared to the original DDPG paper.
        net = dense(net, 64)

        # Add final output layer to produce action values (Q values)
        Q_values = layers.Dense(units=1, name='q_values')(net)

        # Create Keras model
        self.model = models.Model(inputs=[states, actions], outputs=Q_values)
        print('Critic model')
        self.model.summary()

        # Define optimizer and compile model for training
        # with built-in loss function
        optimizer = optimizers.Adam(lr=self.learning_rate)
        self.model.compile(optimizer=optimizer, loss='mse')

        # Compute action gradients (derivative of Q values w.r.t. to actions)
        action_gradients = K.gradients(Q_values, actions)

        # Define an additional function to fetch action gradients
        # (to be used by actor model)
        self.get_action_gradients = K.function(
            inputs=[*self.model.input, K.learning_phase()],
            outputs=action_gradients
        )
