"""Solution."""
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, ConstantKernel


# global variables
DOMAIN = np.array([[0, 10]])  # restrict \theta in [0, 10]
SAFETY_THRESHOLD = 4  # threshold, upper bound of SA

sigma_f = 0.15
sigma_v = 0.0001



# TODO: implement a self-contained solution in the BOAlgorithm class.
# NOTE: main() is not called by the checker.
# The domain is X=[0,10]
# .
# The noise perturbing the observation is Gaussian with standard deviation σf=0.15
# and σv=0.0001
# for logP and SA, respectively.
# The mapping f
# can be effectively modeled with a Matérn with smoothness parameter ν=2.5 or a RBF kernel with variance 0.5, lengthscale 10, 1 or 0.5
# . To achieve the best result, we recommend tuning the kernel and the lengthscale parameter.
# The mapping v
# can be effectively modeled with an additive kernel composed of a Linear kernel and a Matérn with smoothness parameter ν=2.5 or a RBF kernel with variance √2, lengthscale 10, 1 or 0.5. And the prior mean should be 4
# . To achieve the best result, we recommend tuning the kernel and the lengthscale parameter.
# The maximum tolerated SA is κ=4
# .
class BOAlgorithm():
    def __init__(self):
        """Initializes the algorithm with a parameter configuration."""
        self.X = []
        self.Y_f = []
        self.Y_v = []

        sigma_f = 0.15 
        sigma_v = 0.0001
        
        kernels_f = [
            Matern(length_scale=10, nu=2.5),
            Matern(length_scale=1, nu=2.5),
            Matern(length_scale=0.5, nu=2.5),
            RBF(length_scale=10),
            RBF(length_scale=1),
            RBF(length_scale=0.5),
        ]

        kernels_v = [
            ConstantKernel(4) + Matern(length_scale=10, nu=2.5)*(2**0.5),
            ConstantKernel(4) + Matern(length_scale=1, nu=2.5)*(2**0.5),
            ConstantKernel(4) + Matern(length_scale=0.5, nu=2.5)*(2**0.5),
            ConstantKernel(4) + RBF(length_scale=10)*(2**0.5),
            ConstantKernel(4) + RBF(length_scale=1)*(2**0.5),
            ConstantKernel(4) + RBF(length_scale=0.5)*(2**0.5),
        ]

        self.gp_f = GaussianProcessRegressor(
            kernel=kernels_f[1],
            alpha=sigma_f**2,
            optimizer=None
        )

        self.gp_v = GaussianProcessRegressor(
            kernel=kernels_v[1],
            alpha=sigma_v**2,
            optimizer=None
        )

    def recommend_next(self):
        """
        Recommend the next input to sample.

        Returns
        -------
        recommendation: float
            the next point to evaluate
        """
        X = np.array(self.X).reshape(-1, 1)
        Y_f = np.array(self.Y_f).reshape(-1, 1)
        Y_v = np.array(self.Y_v).reshape(-1, 1)

        self.gp_f.fit(X, Y_f)
        self.gp_v.fit(X, Y_v)

        return self.optimize_acquisition_function()

    def optimize_acquisition_function(self):
        """Optimizes the acquisition function defined below (DO NOT MODIFY).

        Returns
        -------
        x_opt: float
            the point that maximizes the acquisition function, where
            x_opt in range of DOMAIN
        """

        def objective(x):
            return -self.acquisition_function(x)

        f_values = []
        x_values = []

        # Restarts the optimization 20 times and pick the best solution
        for _ in range(20):
            x0 = DOMAIN[:, 0] + (DOMAIN[:, 1] - DOMAIN[:, 0]) * \
                 np.random.rand(DOMAIN.shape[0])
            result = fmin_l_bfgs_b(objective, x0=x0, bounds=DOMAIN,
                                   approx_grad=True)
            x_values.append(np.clip(result[0], *DOMAIN[0]))
            f_values.append(-result[1])

        ind = np.argmax(f_values)
        x_opt = x_values[ind].item()

        return x_opt

    def acquisition_function(self, x: np.ndarray):
        """Compute the acquisition function for x.

        Parameters
        ----------
        x: np.ndarray
            x in domain of f, has shape (N, 1)

        Returns
        ------
        af_value: np.ndarray
            shape (N, 1)
            Value of the acquisition function at x
        """
        x = np.atleast_2d(x)
        
        mean_f, std_f = self.gp_f.predict(x, return_std=True)
        mean_v, std_v = self.gp_v.predict(x, return_std=True)
        
        deviation_count = 3
        penalty = 1.5
        ucb = mean_v + deviation_count * std_v
        if ucb > SAFETY_THRESHOLD:
            diff = ucb - SAFETY_THRESHOLD
            penalty *= diff
            return mean_f - penalty
        else:
            return mean_f + std_f


    def add_observation(self, x: float, f: float, v: float):
        """
        Add data points to the model.

        Parameters
        ----------
        x: float
            structural features
        f: float
            logP obj func
        v: float
            SA constraint func
        """
        self.X.append(x)
        self.Y_f.append(f)
        self.Y_v.append(v)

    def get_optimal_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: float
            the optimal solution of the problem
        """
        X = np.array(self.X).reshape(-1, 1)
        Y_f = np.array(self.Y_f).reshape(-1, 1)
        Y_v = np.array(self.Y_v).reshape(-1, 1)

        safe_indices = np.where(Y_v < SAFETY_THRESHOLD)[0]
        if len(safe_indices) == 0:
            raise ValueError("No safe points found.")

        safe_X = X[safe_indices]
        safe_Y_f = Y_f[safe_indices]

        best_safe_index = np.argmax(safe_Y_f)
        best_safe_point = safe_X[best_safe_index]

        return best_safe_point.item()


    def plot(self, plot_recommendation: bool = True):
        """Plot objective and constraint posterior for debugging (OPTIONAL).

        Parameters
        ----------
        plot_recommendation: bool
            Plots the recommended point if True.
        """
        pass


# ---
# TOY PROBLEM. To check your code works as expected (ignored by checker).
# ---

def check_in_domain(x: float):
    """Validate input"""
    x = np.atleast_2d(x)
    return np.all(x >= DOMAIN[None, :, 0]) and np.all(x <= DOMAIN[None, :, 1])


def f(x: float):
    """Dummy logP objective"""
    mid_point = DOMAIN[:, 0] + 0.5 * (DOMAIN[:, 1] - DOMAIN[:, 0])
    return - np.linalg.norm(x - mid_point, 2)


def v(x: float):
    """Dummy SA"""
    return 2.0


def get_initial_safe_point():
    """Return initial safe point"""
    x_domain = np.linspace(*DOMAIN[0], 4000)[:, None]
    print(x_domain)
    c_val = np.vectorize(v)(x_domain)
    print(c_val)
    x_valid = x_domain[c_val < SAFETY_THRESHOLD]
    np.random.seed(0)
    np.random.shuffle(x_valid)
    x_init = x_valid[0]

    return x_init


def main():
    """FOR ILLUSTRATION / TESTING ONLY (NOT CALLED BY CHECKER)."""
    # Init problem
    agent = BOAlgorithm()

    # Add initial safe point
    x_init = get_initial_safe_point()
    obj_val = f(x_init)
    cost_val = v(x_init)
    print(f'Initial safe point {x_init} with obj {obj_val} and cost {cost_val}')
    agent.add_observation(x_init, obj_val, cost_val)

    # Loop until budget is exhausted
    for j in range(20):
        # Get next recommendation
        x = agent.recommend_next()

        # Check for valid shape
        assert x.shape == (1, DOMAIN.shape[0]), \
            f"The function recommend_next must return a numpy array of " \
            f"shape (1, {DOMAIN.shape[0]})"

        # Obtain objective and constraint observation
        obj_val = f(x) + np.randn()
        cost_val = v(x) + np.randn()
        agent.add_observation(x, obj_val, cost_val)

    # Validate solution
    solution = agent.get_optimal_solution()
    assert check_in_domain(solution), \
        f'The function get_optimal_solution must return a point within the' \
        f'DOMAIN, {solution} returned instead'

    # Compute regret
    regret = (0 - f(solution))

    print(f'Optimal value: 0\nProposed solution {solution}\nSolution value '
          f'{f(solution)}\nRegret {regret}\nUnsafe-evals TODO\n')


if __name__ == "__main__":
    main()
