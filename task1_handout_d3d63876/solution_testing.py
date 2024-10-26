import os
import typing
from sklearn.gaussian_process.kernels import *
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
import matplotlib.pyplot as plt
from matplotlib import cm

import pickle
import time


# Set `EXTENDED_EVALUATION` to `True` in order to visualize your predictions.
EXTENDED_EVALUATION = True
EVALUATION_GRID_POINTS = 300  # Number of grid points used in extended evaluation

# Cost function constants
COST_W_UNDERPREDICT = 50.0
COST_W_NORMAL = 1.0

# block coordinates
# BLOCK_COORDS = [[0.0, 0.0, 0.2, 0.2], [0.2, 0, 0.55, 0.3], [0.55, 0.0, 1.0, 0.3], 
#                 [0.0, 0.2, 0.2, 1.0], [0.2, 0.3, 0.55, 0.45], 
#                 [0.55, 0.3, 0.8, 0.45], [0.8, 0.3, 1.0, 0.45],
#                 [0.2, 0.45, 0.55, 0.6], [0.2, 0.6, 0.55, 0.8], [0.2, 0.8, 0.55, 1.0], 
#                 [0.55, 0.45, 0.75, 0.75], [0.55, 0.75, 0.75, 1.0], # Divide this asymmetrically
#                 [0.75, 0.45, 1.0, 1.0],
#                 ]
# Modified block coordinates
BLOCK_COORDS = [[0.0, 0.0, 0.2, 0.2], [0.2, 0, 0.55, 0.3], [0.55, 0.0, 1.0, 0.3], 
                [0.0, 0.2, 0.2, 0.65], [0.0, 0.65, 0.2, 1.0], [0.2, 0.3, 0.55, 0.45], 
                [0.55, 0.3, 0.8, 0.45], [0.8, 0.3, 1.0, 0.45],
                [0.2, 0.45, 0.55, 0.6], [0.2, 0.6, 0.55, 0.8], [0.2, 0.8, 0.4, 1.0], [0.4, 0.8, 0.55, 1.0], 
                [0.55, 0.45, 0.75, 0.75], [0.55, 0.75, 0.75, 1.0], # Divide this asymmetrically
                [0.75, 0.45, 1.0, 0.75], [0.75, 0.75, 1.0, 1.0]
                ]

class Model(object):
    """
    Model for this task.
    You need to implement the train_model and generate_predictions methods
    without changing their signatures, but are allowed to create additional methods.
    """

    def __init__(self):
        """
        Initialize your model here.
        We already provide a random number generator for reproducibility.
        """
        self.rng = np.random.default_rng(seed=0)

        # TODO: Add custom initialization for your model here if necessary
        self.NUM_GP = len(BLOCK_COORDS)
        self.kernel = Matern(length_scale=0.01, length_scale_bounds=(1e-10, 1e6), nu=2.5) + RBF(length_scale=10, length_scale_bounds=(1e-10, 1e6)) + WhiteKernel(noise_level_bounds=(1e-10, 1e4)) 
        self.globalGP = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=10)
        self.GPs = []
        for i in range(self.NUM_GP):
            self.GPs.append(GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=10, alpha=1e-10 ,normalize_y=True))

        self.block_coordinates = BLOCK_COORDS

    def generate_predictions(self, test_coordinates: np.ndarray, test_area_flags: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict the pollution concentration for a given set of city_areas.
        :param test_coordinates: city_areas as a 2d NumPy float array of shape (NUM_SAMPLES, 2)
        :param test_area_flags: city_area info for every sample in a form of a bool array (NUM_SAMPLES,)
        :return:
            Tuple of three 1d NumPy float arrays, each of shape (NUM_SAMPLES,),
            containing your predictions, the GP posterior mean, and the GP posterior stddev (in that order)
        """

        # TODO: Use your GP to estimate the posterior mean and stddev for each city_area here
        gp_mean = np.zeros(test_coordinates.shape[0], dtype=float)
        gp_std = np.zeros(test_coordinates.shape[0], dtype=float)
        predictions = np.zeros(test_coordinates.shape[0], dtype=float)
        area_gp_mean = np.zeros(test_area_flags.shape[0], dtype=float)
        area_gp_std = np.zeros(test_area_flags.shape[0], dtype=float)

        # TODO: Use the GP posterior to form your predictions here
        # gp_mean, gp_std = self.GP.predict(test_coordinates, return_std=True)
        # predictions = gp_mean

        for i in range(self.NUM_GP):
            # Define the block boundaries
            block_min = self.block_coordinates[i][0:2]
            block_max = self.block_coordinates[i][2:]

            # Find the indices of the points within the block
            in_block = np.all((test_coordinates >= block_min) & (test_coordinates < block_max), axis=1)
            block_indices = np.where(in_block)[0]

            gp_mean[block_indices], gp_std[block_indices] = self.GPs[i].predict(test_coordinates[block_indices], return_std=True)

        predictions = gp_mean
        print(gp_std)
        print(predictions)
        # Add the residential area cost to prevent underestimation
        test_area_flags = test_area_flags.astype(bool)
        # area_gp_mean, area_gp_std = self.globalGP.predict(test_coordinates[test_area_flags], return_std=True)
        # predictions[test_area_flags] = area_gp_mean + 1 * area_gp_std
        predictions[test_area_flags] = gp_mean[test_area_flags] + 0.95 * gp_std[test_area_flags]
        print(predictions)

        return predictions, gp_mean, gp_std

    def train_model(self, train_targets: np.ndarray, train_coordinates: np.ndarray, train_area_flags: np.ndarray):
        """
        Fit your model on the given training data.
        :param train_coordinates: Training features as a 2d NumPy float array of shape (NUM_SAMPLES, 2)
        :param train_targets: Training pollution concentrations as a 1d NumPy float array of shape (NUM_SAMPLES,)
        :param train_area_flags: Binary variable denoting whether the 2D training point is in the residential area (1) or not (0)
        """

        # TODO: Fit your model here

        # randomly sample the training data
        # num_samples = 2000
        # sample_indices = self.rng.choice(num_samples, size=num_samples, replace=False)
        # train_coordinates = train_coordinates[sample_indices]
        # train_targets = train_targets[sample_indices]

        # Check if 'gp_model.pkl' exists
        start_time = time.time()

        # if os.path.exists('global_GP.pkl'):  
        #     with open('global_GP.pkl', 'rb') as file:
        #         self.globalGP = pickle.load(file)
        # else: 
        #     # Fit a global GP using only samples from sensitive areas
        # mask_residential_are = train_area_flags == 1 
        # self.globalGP.fit(train_coordinates[mask_residential_are], train_targets[mask_residential_are])
            # # Save the GP model to a file using pickle
            # with open('global_GP.pkl', 'wb') as file:
            #     pickle.dump(self.globalGP, file)

        elapsed_time = time.time() - start_time
        print(f"Elapsed time for fitting/loading global GP: {elapsed_time:.2f} seconds")
        

        # Divide the training data into blocks and sample from those blocks
        MAX_SAMPLES_IN_EACH_BLOCK = 2500
        samples_per_block = []
        print(f"Max samples in each block: {MAX_SAMPLES_IN_EACH_BLOCK}")
        for i in range(len(self.block_coordinates)):
            # Define the block boundaries
            block_min = self.block_coordinates[i][0:2]
            block_max = self.block_coordinates[i][2:]

            # Find the indices of the points within the block
            in_block = np.all((train_coordinates >= block_min) & (train_coordinates < block_max), axis=1)

            # Sample uniformly from the points within the block
            block_indices = np.where(in_block)[0]
            print(f'There are {len(block_indices)} samples in {i+1}th block.')
            if len(block_indices) > MAX_SAMPLES_IN_EACH_BLOCK: #IDEA: again divide into blocks and sample uniformly
                sample_index = self.rng.choice(block_indices, size=MAX_SAMPLES_IN_EACH_BLOCK)
                samples_per_block.append([train_coordinates[sample_index], train_targets[sample_index]])
            else:
                samples_per_block.append([train_coordinates[block_indices], train_targets[block_indices]])

        # Fit the GP model on the sampled data
        for i in range(self.NUM_GP):
            print(f'Fitting {i+1}th GP out of {self.NUM_GP} GPs')
            self.GPs[i].fit(samples_per_block[i][0], samples_per_block[i][1])
        
        # self.GP.fit(train_coordinates, train_targets)
        return


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


def visualize_train_data(train_coordinates: np.ndarray, train_targets: np.ndarray, train_area_flags: np.ndarray, output_dir: str = './results'):
    """
    Visualizes the training data.
    :param train_coordinates: Training features as a 2d NumPy float array of shape (NUM_SAMPLES, 2)
    :param train_targets: Training pollution concentrations as a 1d NumPy float array of shape (NUM_SAMPLES,)
    :param train_area_flags: Binary variable denoting whether the 2D training point is in the residential area (1) or not (0)
    :param output_dir: Directory in which the visualizations will be stored
    """
    print('Visualizing training data')

    # Create a scatter plot of the training data (Pollution Level)
    fig1, ax1 = plt.subplots()
    scatter1 = ax1.scatter(train_coordinates[:, 0], train_coordinates[:, 1], c=train_targets, cmap='viridis', s=10, label='Pollution Level')
    ax1.set_title('Visualization of training data (Pollution Level)')
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    cbar1 = fig1.colorbar(scatter1, ax=ax1)
    cbar1.set_label('Pollution Level')

    # Save figure to pdf
    os.makedirs(output_dir, exist_ok=True)
    figure_path1 = os.path.join(output_dir, 'train_data_pollution_level.pdf')
    fig1.savefig(figure_path1)
    print(f'Saved training data visualization (Pollution Level) to {figure_path1}')

    # Create a scatter plot of the training data (Residential Area)
    fig2, ax2 = plt.subplots()
    residential_mask = train_area_flags.astype(bool)
    scatter2 = ax2.scatter(train_coordinates[:, 0], train_coordinates[:, 1], c=residential_mask, cmap='coolwarm', s=10, label='Residential Area')
    ax2.set_title('Visualization of training data (Residential Area)')
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')
    cbar2 = fig2.colorbar(scatter2, ax=ax2)
    cbar2.set_label('Residential Area')

    # Save figure to pdf
    figure_path2 = os.path.join(output_dir, 'train_data_residential_area.pdf')
    fig2.savefig(figure_path2)
    print(f'Saved training data visualization (Residential Area) to {figure_path2}')

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
    test_coordinates = test_x[:, :2]
    train_area_flags = train_x[:, 2].astype(bool)
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

    # Print data information
    print(f'Training data shape: {train_x.shape}')
    print(f'Mean and Standard deviation of pollution level: {np.mean(train_y)}, {np.std(train_y)}')
    print(f'Max and Min of pollution level: {np.max(train_y)}, {np.min(train_y)}')
    print(f'Mean and Standard deviation of coordinates: {np.mean(train_coordinates, axis=0)}, {np.std(train_coordinates, axis=0)}')
    print(f'Max and Min of coordinates: {np.max(train_coordinates, axis=0)}, {np.min(train_coordinates, axis=0)}')
    print(f'Number of residential area samples: {np.sum(train_area_flags)}')
    print('')
    print(f'Test data shape: {test_x.shape}')
    print(f'Test coordinates shape: {test_coordinates.shape}')
    print(f'Test area flags shape: {test_area_flags.shape}')
    
    # Fit the model
    print('Training model')
    model = Model()
    model.train_model(train_y, train_coordinates, train_area_flags)

    # Predict on the test features
    print('Predicting on test features')
    predictions = model.generate_predictions(test_coordinates, test_area_flags)
    print(predictions)

    # visualize_train_data(train_coordinates, train_y, train_area_flags)

    if EXTENDED_EVALUATION:
        execute_extended_evaluation(model, output_dir='.')


if __name__ == "__main__":
    main()
