import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

def pruefe_gewichte(nn, gewicht1, gewicht2, gewicht3, gewicht4):
    """
    Parameters:
    nn: A network object with attributes fc1, fc2, fc3, whose parameters are NumPy arrays.
    gewicht1: Expected value for fc1.weight[0][0]
    gewicht2: Expected value for fc1.bias[4]
    gewicht3: Expected value for fc2.bias[1]
    gewicht4: Expected value for fc3.weight[2][3]
    
    Returns:
      A string summarizing whether each weight has been read correctly.
    """
    # For instance, nn.fc1.weight is assumed to be a NumPy array.
    g1 = round(nn.fc1.weight[0][0], 4)
    g2 = round(nn.fc1.bias[4], 4)
    g3 = round(nn.fc2.bias[1], 4)
    g4 = round(nn.fc3.weight[2][3], 4)

    wrong = False
    result = ""

    if g1 == round(gewicht1, 4):
        result = "Das erste Gewicht hast du richtig abgelesen!\n"
    else:
        result = "Das erste Gewicht hast du nicht richtig abgelesen!\n"
        wrong = True

    if g2 == round(gewicht2, 4):
        result += "Das zweite Gewicht hast du richtig abgelesen!\n"
    else:
        result += "Das zweite Gewicht hast du nicht richtig abgelesen!\n"
        wrong = True

    if g3 == round(gewicht3, 4):
        result += "Das dritte Gewicht hast du richtig abgelesen!\n"
    else:
        result += "Das dritte Gewicht hast du nicht richtig abgelesen!\n"
        wrong = True 

    if g4 == round(gewicht4, 4):
        result += "Das vierte Gewicht hast du richtig abgelesen!\n"
    else:
        result += "Das vierte Gewicht hast du nicht richtig abgelesen!\n"
        wrong = True

    if wrong:
        return result
    else:
        return "Super! Du hast alle Gewichte richtig abgelesen!"


def daten():
    """
    Returns:
    x_train: A NumPy array of training input data of shape (400, 2)
    y_train: A NumPy array of training labels of shape (400,)
    x_test:  A NumPy array of test input data of shape (100, 2)
    y_test:  A NumPy array of test labels of shape (100,)
    """
    # Create base data (all ones)
    n_data_train = np.ones((200, 2))
    n_data_test  = np.ones((50, 2))

    # Create training data:
    # Class 0: Normal distribution centered at [2.5, 5] (for all 200 samples)
    x0 = np.random.normal(loc=n_data_train + np.array([2.5, 5]), scale=1.0)
    y0 = np.zeros(200, dtype=int)
    # Class 1: Normal distribution centered at [8, 2]
    x1 = np.random.normal(loc=n_data_train + np.array([8, 2]), scale=1.0)
    y1 = np.ones(200, dtype=int)

    x_train = np.concatenate((x0, x1), axis=0).astype(np.float32)
    y_train = np.concatenate((y0, y1), axis=0)

    # Create test data:
    n_data_test = np.ones((50, 2))
    x0_test = np.random.normal(loc=n_data_test + np.array([2.5, 5]), scale=1.0)
    y0_test = np.zeros(50, dtype=int)
    x1_test = np.random.normal(loc=n_data_test + np.array([8, 2]), scale=1.0)
    y1_test = np.ones(50, dtype=int)

    x_test = np.concatenate((x0_test, x1_test), axis=0).astype(np.float32)
    y_test = np.concatenate((y0_test, y1_test), axis=0)

    return x_train, y_train, x_test, y_test
  
def datenpunkte_zeichnen(x_data, labels, farben):
    """
    Parameters:
      x_data: A NumPy array with the data points, shape (N, 2)
      labels: A NumPy array with the class labels of each data point.
      farben: A list of color names for the classes.
    """
    plt.scatter(x_data[:, 0], x_data[:, 1], c=labels, s=50, cmap=colors.ListedColormap(farben))
    plt.show()

    
def daten2():
    """
    Returns:
    x_train: A NumPy array of training input data of shape (600, 2)
    y_train: A NumPy array of training labels of shape (600,)
    x_test:  A NumPy array of test input data of shape (150, 2)
    y_test:  A NumPy array of test labels of shape (150,)`
    """
    # Training data:
    n_data_train = np.ones((200, 2))

    x0 = np.random.normal(loc=n_data_train + np.array([6, 5]), scale=1.0)
    y0 = np.zeros(200, dtype=int)
    x1 = np.random.normal(loc=n_data_train + np.array([2, 2]), scale=1.0)
    y1 = np.ones(200, dtype=int)
    x2 = np.random.normal(loc=n_data_train + np.array([10, 2]), scale=1.0)
    y2 = 2 * np.ones(200, dtype=int)

    x_train = np.concatenate((x0, x1, x2), axis=0).astype(np.float32)
    y_train = np.concatenate((y0, y1, y2), axis=0)

    # Test data:
    n_data_test = np.ones((50, 2))

    x0_test = np.random.normal(loc=n_data_test + np.array([6, 5]), scale=1.0)
    y0_test = np.zeros(50, dtype=int)
    x1_test = np.random.normal(loc=n_data_test + np.array([2, 2]), scale=1.0)
    y1_test = np.ones(50, dtype=int)
    x2_test = np.random.normal(loc=n_data_test + np.array([10, 2]), scale=1.0)
    y2_test = 2 * np.ones(50, dtype=int)

    x_test = np.concatenate((x0_test, x1_test, x2_test), axis=0).astype(np.float32)
    y_test = np.concatenate((y0_test, y1_test, y2_test), axis=0)

    return x_train, y_train, x_test, y_test

