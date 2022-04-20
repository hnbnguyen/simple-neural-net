import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        return nn.DotProduct(self.w, x)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        return 1 if (nn.as_scalar(self.run(x)) >= 0) else -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        #initialize a batch size of 1 for all training run
        batch_size = 1

        flag = True
        while flag:
            flag = False
            # checking the result from the current parameters 
            for a, b in dataset.iterate_once(batch_size):
                result = self.get_prediction(a)
                # not yet converge, continue the while loop
                if result != nn.as_scalar(b):
                    self.w.update(nn.as_scalar(b), a)
                    flag = True

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        # based on the 3-layer deeper network architecture in Neural Network Tips
        self.learning_rate = 0.01

        self.w1 = nn.Parameter(1, 64)
        self.b1 = nn.Parameter(1, 64)

        self.w2 = nn.Parameter(64, 128)
        self.b2 = nn.Parameter(1, 128)
        # output layer
        self.w3 = nn.Parameter(128, 1)
        self.b3 = nn.Parameter(1, 1)

        self.params = [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        # Layer 1
        input = x
        input_lin = nn.Linear(input, self.w1)
        input_bias = nn.AddBias(input_lin, self.b1)
        output_1 = nn.ReLU(input_bias)

        # Layer 2
        input_lin2 = nn.Linear(output_1, self.w2)
        input_bias2 = nn.AddBias(input_lin2, self.b2)
        output_2 = nn.ReLU(input_bias2)

        # Layer 3 (output)
        input_lin3 = nn.Linear(output_2, self.w3)
        output_3 = nn.AddBias(input_lin3, self.b3)

        return output_3

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        predictions = self.run(x)
        return nn.SquareLoss(predictions,y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        batch_size = 50
        getloss = float('inf')

        # Benchmark of loss is 0.2
        while getloss >= 0.01:
            # For each component in the node 
            for x, y in dataset.iterate_once(batch_size):
                # Compute loss for the batch
                getloss = self.get_loss(x, y)
                # Obtain the value of the node as a scalar number
                getloss = nn.as_scalar(getloss)

                # Obtain the gradient of the loss with respect to the parameters
                gradients = nn.gradients(self.params, getloss)

                # Update the learning rate
                for i in range(len(self.params)):
                    self.params[i].update(-self.learning_rate, gradients[i])

class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.learning_rate = 0.1

        self.w1 = nn.Parameter(784, 256)
        self.b1 = nn.Parameter(1, 256)

        self.w2 = nn.Parameter(256, 128)
        self.b2 = nn.Parameter(1, 128)

        self.w3 = nn.Parameter(128, 64)
        self.b3 = nn.Parameter(1, 64)

        self.w4 = nn.Parameter(64, 10)
        self.b4 = nn.Parameter(1, 10)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"

        #Hidden Layer One
        trans_1 = nn.Linear(x, self.w1)
        trans_bias1 = nn.AddBias(trans_1, self.b1)
        layer1 = nn.ReLU(trans_bias1)

        #Hidden Layer Two
        trans_2 = nn.Linear(layer1, self.w2)
        trans_bias2 = nn.AddBias(trans_2, self.b2)
        layer2 = nn.ReLU(trans_bias2)

        #Hidden Layer Three
        trans_3 = nn.Linear(layer2, self.w3)
        trans_bias3 = nn.AddBias(trans_3, self.b3)
        layer3 = nn.ReLU(trans_bias3)

        #Output vector without ReLu
        output_layer = nn.AddBias(nn.Linear(layer3, self.w4), self.b4)

        #Return the output layer
        return output_layer

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"

        loss_node = self.run(x)
        return nn.SoftmaxLoss(loss_node, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        
        #Setting accuracy 
        validAccuracy = 0

        #Creating batch size
        batchSize = 85

        #Creating arbitrary loss value (to be changed)
        loss = float(1000000)

        #iterate until sufficient accuracy 
        while validAccuracy < .98:

            #getting x and y values to determine loss 
            for x, y in dataset.iterate_once(batchSize):

                #operations
                loss = self.get_loss(x, y)
                #determining gradient loss
                grads = nn.gradients(self.params, loss)
                #geting as a scalar
                loss = nn.as_scalar(loss)

                #iterating over the parameters
                for i in range(len(self.params)):

                    #updating the parameters 
                    self.params[i].update(-self.learning_rate, grads[i])

            #calculating accurary
            validAccuracy = dataset.get_validation_accuracy()

