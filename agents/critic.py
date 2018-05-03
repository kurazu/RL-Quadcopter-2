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
        net_states = dense(states, 64)
        # net_states = dense(
        #     net_states, 400, activation='relu', batch_normalization=True
        # )

        # Add hidden layer(s) for action pathway
        net_actions = dense(actions, 32)
        # net_actions = dense(
        #     net_actions, 400, activation='relu', batch_normalization=True
        # )

        # Try different layer sizes, activations,
        # add batch normalization, regularizers, etc.

        # Combine state and action pathways
        net = layers.Add()([net_states, net_actions])

        # Add more layers to the combined network if needed
        net = dense(net, 64)
        # net = dense(net, 400)

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
