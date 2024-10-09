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
        return nn.DotProduct(x, self.get_weights())

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        score = self.run(x)
        if nn.as_scalar(score) >= 0:
            return 1
        else:
            return -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        batch_size = 1
        while True:
            covr = True
            for x, y in dataset.iterate_once(batch_size):
                pre = self.get_prediction(x)
                if pre != nn.as_scalar(y):
                    covr = False
                    self.w.update(x, nn.as_scalar(y))
            if covr:
                break

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.input_size = 1
        self.output_size = 1
        self.hidden_layer_size = 512
        self.batch_size = 50
        self.learning_rate = 0.05

        self.w1 = nn.Parameter(self.input_size, self.hidden_layer_size)
        self.b1 = nn.Parameter(1, self.hidden_layer_size)
        self.w2 = nn.Parameter(self.hidden_layer_size, self.output_size)
        self.b2 = nn.Parameter(1, self.output_size)
        self.param = [self.w1, self.b1, self.w2, self.b2]

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        layer = nn.ReLU(nn.AddBias(nn.Linear(x, self.w1), self.b1))
        predic = nn.AddBias(nn.Linear(layer, self.w2), self.b2)
        return predic

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
        loss = nn.SquareLoss(self.run(x), y)
        return loss

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        loss  = float('inf')

        while loss > 0.01:
            for x, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x, y)
                gradient_w1, gradient_b1, gradient_w2, gradient_b2 = nn.gradients(loss, self.param)
                loss = nn.as_scalar(loss)
                self.w1.update(gradient_w1, -self.learning_rate)
                self.b1.update(gradient_b1, -self.learning_rate)
                self.w2.update(gradient_w2, -self.learning_rate)
                self.b2.update(gradient_b2, -self.learning_rate)
                self.param = [self.w1, self.b1, self.w2, self.b2]

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
        self.input_size = 784
        self.hidden_layer_size1 = 500
        self.hidden_layer_size2 = 250
        self.hidden_layer_size3 = 128
        self.hidden_layer_size4 = 64
        self.output_size = 10
        self.batch_size = 100
        self.learning_rate = 0.1

        self.w1 = nn.Parameter(self.input_size, self.hidden_layer_size1)
        self.b1 = nn.Parameter(1, self.hidden_layer_size1)
        self.w2 = nn.Parameter(self.hidden_layer_size1, self.hidden_layer_size2)
        self.b2 = nn.Parameter(1, self.hidden_layer_size2)
        self.w3 = nn.Parameter(self.hidden_layer_size2, self.hidden_layer_size3)
        self.b3 = nn.Parameter(1, self.hidden_layer_size3)
        self.w4 = nn.Parameter(self.hidden_layer_size3, self.hidden_layer_size4)
        self.b4 = nn.Parameter(1, self.hidden_layer_size4)
        self.w5 = nn.Parameter(self.hidden_layer_size4, self.output_size)
        self.b5 = nn.Parameter(1, self.output_size)
        self.param = [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3, self.w4, self.b4, self.w5, self.b5]

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
        layer1 = nn.ReLU(nn.AddBias(nn.Linear(x, self.w1), self.b1))
        layer2 = nn.ReLU(nn.AddBias(nn.Linear(layer1, self.w2), self.b2))
        layer3 = nn.ReLU(nn.AddBias(nn.Linear(layer2, self.w3), self.b3))
        layer4 = nn.ReLU(nn.AddBias(nn.Linear(layer3, self.w4), self.b4))
        prediction = nn.AddBias(nn.Linear(layer4, self.w5), self.b5)
        return prediction

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
        r = self.run(x)
        loss = nn.SoftmaxLoss(r, y)
        return loss

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        acc = 0
        while acc < 0.98:
            for x, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x, y)
                gradient_w1, gradient_b1, gradient_w2, gradient_b2, gradient_w3, gradient_b3, gradient_w4, gradient_b4, gradient_w5, gradient_b5 = nn.gradients(loss, self.param)
                loss = nn.as_scalar(loss)
                self.w1.update(gradient_w1, -self.learning_rate)
                self.b1.update(gradient_b1, -self.learning_rate)
                self.w2.update(gradient_w2, -self.learning_rate)
                self.b2.update(gradient_b2, -self.learning_rate)
                self.w3.update(gradient_w3, -self.learning_rate)
                self.b3.update(gradient_b3, -self.learning_rate)
                self.w4.update(gradient_w4, -self.learning_rate)
                self.b4.update(gradient_b4, -self.learning_rate)
                self.w5.update(gradient_w5, -self.learning_rate)
                self.b5.update(gradient_b5, -self.learning_rate)
                self.param = [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3, self.w4, self.b4, self.w5, self.b5]
            acc = dataset.get_validation_accuracy()


class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.input_size = 47
        self.hidden_layer_size1 = 380
        self.hidden_layer_size2 = 380
        self.hidden_layer_size3 = 25
        self.output_size = 5
        self.batch_size = 100
        self.learning_rate = 0.1

        self.w1 = nn.Parameter(self.input_size, self.hidden_layer_size1)
        self.b1 = nn.Parameter(1, self.hidden_layer_size1)
        self.w2 = nn.Parameter(self.hidden_layer_size1, self.hidden_layer_size2)
        self.b2 = nn.Parameter(1, self.hidden_layer_size2)
        self.w3 = nn.Parameter(self.hidden_layer_size2, self.output_size)
        self.b3 = nn.Parameter(1, self.output_size)
        self.param = [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        layer = nn.Linear(nn.DataNode(xs[0].data), self.w1)
        for x in xs:
            layer = nn.ReLU(nn.AddBias(nn.Linear(nn.Add(nn.Linear(x, self.w1), layer), self.w2), self.b2))
        prediction = nn.AddBias(nn.Linear(layer, self.w3), self.b3)
        return prediction

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        r = self.run(xs)
        loss = nn.SoftmaxLoss(r, y)
        return loss

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        acc = 0
        while acc < 0.81:
            for x, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x, y)
                gradient_w1, gradient_b1, gradient_w2, gradient_b2, gradient_w3, gradient_b3 = nn.gradients(
                    loss, self.param)
                loss = nn.as_scalar(loss)
                self.w1.update(gradient_w1, -self.learning_rate)
                self.b1.update(gradient_b1, -self.learning_rate)
                self.w2.update(gradient_w2, -self.learning_rate)
                self.b2.update(gradient_b2, -self.learning_rate)
                self.w3.update(gradient_w3, -self.learning_rate)
                self.b3.update(gradient_b3, -self.learning_rate)
                self.param = [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]
            acc = dataset.get_validation_accuracy()

