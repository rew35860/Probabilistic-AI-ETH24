import os
import typing
from sklearn.gaussian_process.kernels import *
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
import matplotlib.pyplot as plt
from matplotlib import cm

# Additional imports 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import KFold

# Set `EXTENDED_EVALUATION` to `True` in order to visualize your predictions.
EXTENDED_EVALUATION = True
EVALUATION_GRID_POINTS = 300  # Number of grid points used in extended evaluation

# Cost function constants
COST_W_UNDERPREDICT = 50.0
COST_W_NORMAL = 1.0


class Model(object):
    def __init__(self):
        self.rng = np.random.default_rng(seed=0)
        self.model = None  # Will be set after hyperparameter tuning

    def asymmetric_cost(self, y_true, y_pred, area_flags):
        cost = (y_true - y_pred) ** 2
        weights = np.ones_like(cost) * COST_W_NORMAL
        mask = (y_pred < y_true) & (area_flags == 1)
        weights[mask] = COST_W_UNDERPREDICT
        weighted_cost = cost * weights
        mean_cost = np.mean(weighted_cost)
        return mean_cost

    def generate_predictions(self, test_coordinates, test_area_flags):
        # Obtain GP posterior mean and standard deviation
        gp_mean, gp_std = self.model.predict(test_coordinates, return_std=True)
        # Initialize predictions with GP mean
        predictions = gp_mean.copy()
        # Adjust predictions in candidate residential areas
        candidate_area_mask = test_area_flags == 1
        quantile = 1.645  # 95th percentile
        predictions[candidate_area_mask] = gp_mean[candidate_area_mask] + quantile * gp_std[candidate_area_mask]
        return predictions, gp_mean, gp_std

    def train_model(self, train_targets, train_coordinates, train_area_flags):
        sample_indices = self.rng.choice(train_targets.shape[0], 1000, replace=False)
        X = train_coordinates[sample_indices]
        y = train_targets[sample_indices]
        area_flags = train_area_flags[sample_indices]

        # Define candidate kernels
        kernel_options = [
            1.0 * RBF(length_scale=1.0),
            1.0 * Matern(length_scale=1.0, nu=1.5),
            1.0 * RationalQuadratic(length_scale=1.0, alpha=1.0),
            # Add more kernels or combinations if needed
        ]

        # Define parameter grid
        param_grid = {
            "kernel": kernel_options,
            "alpha": [1e-10, 1e-5, 1e-2],  # Noise level
            "optimizer": ["fmin_l_bfgs_b"],
            "n_restarts_optimizer": [0, 5],
        }

        # # Initialize a basic GPR model
        # gpr = GaussianProcessRegressor()

        # Perform custom cross-validation
        # best_params, best_score = self.custom_cross_val_score(
        #     estimator=gpr, X=X, y=y, area_flags=area_flags, cv=3, param_grid=param_grid
        # )

        # Set the best parameters to the model
        self.model = GaussianProcessRegressor(kernel=1**2 * Matern(length_scale=1, nu=1.5), alpha=1e-10, optimizer="fmin_l_bfgs_b", n_restarts_optimizer=5)

        # print(f"Best parameters: {best_params}")
        # print(f"Best score: {best_score}")
        # print(f"Best kernel: {best_params['kernel']}")

        # Fit the model on the entire training data
        self.model.fit(X, y)

    def custom_cross_val_score(self, estimator, X, y, area_flags, cv, param_grid):
        best_score = float('inf')
        best_params = None
        for params in ParameterGrid(param_grid):
            estimator.set_params(**params)
            scores = []
            kf = KFold(n_splits=cv, shuffle=True, random_state=0)
            for train_index, test_index in kf.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                area_flags_train, area_flags_test = area_flags[train_index], area_flags[test_index]
                estimator.fit(X_train, y_train)
                y_pred = estimator.predict(X_test)
                cost = self.asymmetric_cost(y_true=y_test, y_pred=y_pred, area_flags=area_flags_test)
                scores.append(cost)
            avg_score = np.mean(scores)
            if avg_score < best_score:
                best_score = avg_score
                best_params = params.copy()
        return best_params, best_score



# You don't have to change this function
def calculate_cost(ground_truth: np.ndarray, predictions: np.ndarray, area_flags: np.ndarray) -> float:
    """
    Calculates the cost of a set of predictions.

    :param ground_truth: Ground truth pollution levels as a 1d NumPy float array
    :param predictions: Predicted pollution levels as a 1d NumPy float array
    :param area_flags: city_area info for every sample in a form of a bool array (NUM_SAMPLES,)
    :return: Total cost of all predictions as a single float
    """
    assert ground_truth.ndim == 1 and predictions.ndim == 1 and ground_truth.shape == predictions.shape

    # Unweighted cost
    cost = (ground_truth - predictions) ** 2
    weights = np.ones_like(cost) * COST_W_NORMAL

    # Case i): underprediction
    mask = (predictions < ground_truth) & [bool(area_flag) for area_flag in area_flags]
    weights[mask] = COST_W_UNDERPREDICT

    # Weigh the cost and return the average
    return np.mean(cost * weights)


# You don't have to change this function
def check_within_circle(coordinate, circle_parameters):
    """
    Checks if a coordinate is inside a circle.
    :param coordinate: 2D coordinate
    :param circle_parameters: 3D coordinate of the circle center and its radius
    :return: True if the coordinate is inside the circle, False otherwise
    """
    return (coordinate[0] - circle_parameters[0])**2 + (coordinate[1] - circle_parameters[1])**2 < circle_parameters[2]**2

# You don't have to change this function 
def identify_city_area_flags(grid_coordinates):
    """
    Determines the city_area index for each coordinate in the visualization grid.
    :param grid_coordinates: 2D coordinates of the visualization grid
    :return: 1D array of city_area indexes
    """
    # Circles coordinates
    circles = np.array([[0.5488135, 0.71518937, 0.17167342],
                    [0.79915856, 0.46147936, 0.1567626 ],
                    [0.26455561, 0.77423369, 0.10298338],
                    [0.6976312,  0.06022547, 0.04015634],
                    [0.31542835, 0.36371077, 0.17985623],
                    [0.15896958, 0.11037514, 0.07244247],
                    [0.82099323, 0.09710128, 0.08136552],
                    [0.41426299, 0.0641475,  0.04442035],
                    [0.09394051, 0.5759465,  0.08729856],
                    [0.84640867, 0.69947928, 0.04568374],
                    [0.23789282, 0.934214,   0.04039037],
                    [0.82076712, 0.90884372, 0.07434012],
                    [0.09961493, 0.94530153, 0.04755969],
                    [0.88172021, 0.2724369,  0.04483477],
                    [0.9425836,  0.6339977,  0.04979664]])
    
    area_flags = np.zeros((grid_coordinates.shape[0],))

    for i,coordinate in enumerate(grid_coordinates):
        area_flags[i] = any([check_within_circle(coordinate, circ) for circ in circles])

    return area_flags

# You don't have to change this function
def execute_extended_evaluation(model: Model, output_dir: str = '/results'):
    """
    Visualizes the predictions of a fitted model.
    :param model: Fitted model to be visualized
    :param output_dir: Directory in which the visualizations will be stored
    """
    print('Performing extended evaluation')

    # Visualize on a uniform grid over the entire coordinate system
    grid_lat, grid_lon = np.meshgrid(
        np.linspace(0, EVALUATION_GRID_POINTS - 1, num=EVALUATION_GRID_POINTS) / EVALUATION_GRID_POINTS,
        np.linspace(0, EVALUATION_GRID_POINTS - 1, num=EVALUATION_GRID_POINTS) / EVALUATION_GRID_POINTS,
    )
    visualization_grid = np.stack((grid_lon.flatten(), grid_lat.flatten()), axis=1)
    grid_area_flags = identify_city_area_flags(visualization_grid)
    
    # Obtain predictions, means, and stddevs over the entire map
    predictions, gp_mean, gp_stddev = model.generate_predictions(visualization_grid, grid_area_flags)
    predictions = np.reshape(predictions, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))
    gp_mean = np.reshape(gp_mean, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))

    vmin, vmax = 0.0, 65.0

    # Plot the actual predictions
    fig, ax = plt.subplots()
    ax.set_title('Extended visualization of task 1')
    im = ax.imshow(predictions, vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax)

    # Save figure to pdf
    figure_path = os.path.join(output_dir, 'extended_evaluation.pdf')
    fig.savefig(figure_path)
    print(f'Saved extended evaluation to {figure_path}')

    plt.show()

def extract_area_information(train_x: np.ndarray, test_x: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts the city_area information from the training and test features.
    :param train_x: Training features
    :param test_x: Test features
    :return: Tuple of (training features' 2D coordinates, training features' city_area information,
        test features' 2D coordinates, test features' city_area information)
    """
    train_coordinates = np.zeros((train_x.shape[0], 2), dtype=float)
    train_area_flags = np.zeros((train_x.shape[0],), dtype=bool)
    test_coordinates = np.zeros((test_x.shape[0], 2), dtype=float)
    test_area_flags = np.zeros((test_x.shape[0],), dtype=bool)

    #TODO: Extract the city_area information from the training and test features
    train_coordinates = train_x[:, :2]
    train_area_flags = train_x[:, 2].astype(bool)
    test_coordinates = test_x[:, :2]
    test_area_flags = test_x[:, 2].astype(bool)

    assert train_coordinates.shape[0] == train_area_flags.shape[0] and test_coordinates.shape[0] == test_area_flags.shape[0]
    assert train_coordinates.shape[1] == 2 and test_coordinates.shape[1] == 2
    assert train_area_flags.ndim == 1 and test_area_flags.ndim == 1

    return train_coordinates, train_area_flags, test_coordinates, test_area_flags

# you don't have to change this function
def main():
    # Load the training dateset and test features
    train_x = np.loadtxt('train_x.csv', delimiter=',', skiprows=1)
    train_y = np.loadtxt('train_y.csv', delimiter=',', skiprows=1)
    test_x = np.loadtxt('test_x.csv', delimiter=',', skiprows=1)

    # Extract the city_area information
    train_coordinates, train_area_flags, test_coordinates, test_area_flags = extract_area_information(train_x, test_x)
    
    # # Graphing the data to see how it looks like 
    # plt.figure(figsize=(12, 8)) 
    # plt.scatter(train_coordinates[:,0], train_coordinates[:,1], c=train_area_flags, s=5, alpha=0.7)
    # plt.savefig('graphs/raw_data_1.png')

    # Fit the model
    print('Training model')
    model = Model()
    model.train_model(train_y, train_coordinates, train_area_flags)

    # Predict on the test features
    print('Predicting on test features')
    predictions = model.generate_predictions(test_coordinates, test_area_flags)
    print(predictions)

    if EXTENDED_EVALUATION:
        execute_extended_evaluation(model, output_dir='.')


if __name__ == "__main__":
    main()
