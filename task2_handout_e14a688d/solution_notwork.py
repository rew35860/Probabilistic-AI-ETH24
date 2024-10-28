import abc
import collections
import enum
import math
import pathlib
import typing
import warnings

import numpy as np
import torch
import torch.optim
import torch.utils.data
import tqdm
from matplotlib import pyplot as plt

from util import draw_reliability_diagram, cost_function, setup_seeds, calc_calibration_curve

EXTENDED_EVALUATION = True
"""
Set `EXTENDED_EVALUATION` to `True` in order to generate additional plots on validation data.
"""

USE_PRETRAINED_MODEL = True
"""
If `USE_PRETRAINED_MODEL` is `True`, then MAP inference uses provided pretrained weights.
You should not modify MAP training or the CNN architecture before passing the hard baseline.
If you set the constant to `False` (to further experiment),
this solution always performs MAP inference before running your SWAG implementation.
Note that MAP inference can take a long time.
"""


def main():
    raise RuntimeError(
        "This main() method is for illustrative purposes only"
        " and will NEVER be called when running your solution to generate your submission file!\n"
        "The checker always directly interacts with your SWAGInference class and evaluate method.\n"
        "You can remove this exception for local testing, but be aware that any changes to the main() method"
        " are ignored when generating your submission file."
    )

    data_location = pathlib.Path.cwd()
    model_location = pathlib.Path.cwd()
    output_location = pathlib.Path.cwd()

    # Load training data
    training_images = torch.from_numpy(np.load(data_location / "train_xs.npz")["train_xs"])
    training_metadata = np.load(data_location / "train_ys.npz")
    training_labels = torch.from_numpy(training_metadata["train_ys"])
    training_snow_labels = torch.from_numpy(training_metadata["train_is_snow"])
    training_cloud_labels = torch.from_numpy(training_metadata["train_is_cloud"])
    training_dataset = torch.utils.data.TensorDataset(training_images, training_snow_labels, training_cloud_labels, training_labels)

    # Load validation data
    validation_images = torch.from_numpy(np.load(data_location / "val_xs.npz")["val_xs"])
    validation_metadata = np.load(data_location / "val_ys.npz")
    validation_labels = torch.from_numpy(validation_metadata["val_ys"])
    validation_snow_labels = torch.from_numpy(validation_metadata["val_is_snow"])
    validation_cloud_labels = torch.from_numpy(validation_metadata["val_is_cloud"])
    validation_dataset = torch.utils.data.TensorDataset(validation_images, validation_snow_labels, validation_cloud_labels, validation_labels)

    # Fix all randomness
    setup_seeds()

    # Build and run the actual solution
    training_loader = torch.utils.data.DataLoader(
        training_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0,
    )
    swag_inference = SWAGInference(
        train_xs=training_dataset.tensors[0],
        model_dir=model_location,
    )
    swag_inference.fit(training_loader)
    swag_inference.apply_calibration(validation_dataset)

    # fork_rng ensures that the evaluation does not change the rng state.
    # That way, you should get exactly the same results even if you remove evaluation
    # to save computational time when developing the task
    # (as long as you ONLY use torch randomness, and not e.g. random or numpy.random).
    with torch.random.fork_rng():
        evaluate(swag_inference, validation_dataset, EXTENDED_EVALUATION, output_location)


class InferenceType(enum.Enum):
    """
    Inference mode switch for your implementation.
    `MAP` simply predicts the most likely class using pretrained MAP weights.
    `SWAG_DIAGONAL` and `SWAG_FULL` correspond to SWAG-diagonal and the full SWAG method, respectively.
    """
    MAP = 0
    SWAG_DIAGONAL = 1
    SWAG_FULL = 2


class SWAGInference(object):
    """
    Improved implementation of SWAG with enhanced uncertainty estimation and OOD detection.
    """

    def __init__(
        self,
        train_xs: torch.Tensor,
        model_dir: pathlib.Path,
        inference_mode: InferenceType = InferenceType.SWAG_FULL,
        swag_training_epochs: int = 50,  # Increased from 30 to 50
        swag_lr: float = 0.045,
        swag_update_interval: int = 1,
        max_rank_deviation_matrix: int = 20,
        num_bma_samples: int = 50,
    ):
        """
        :param train_xs: Training images (for storage only)
        :param model_dir: Path to directory containing pretrained MAP weights
        :param inference_mode: Control which inference mode (MAP, SWAG-diagonal, full SWAG) to use
        :param swag_training_epochs: Total number of gradient descent epochs for SWAG
        :param swag_lr: Learning rate for SWAG gradient descent
        :param swag_update_interval: Frequency (in epochs) for updating SWAG statistics during gradient descent
        :param max_rank_deviation_matrix: Rank of deviation matrix for full SWAG
        :param num_bma_samples: Number of networks to sample for Bayesian model averaging during prediction
        """

        self.model_dir = model_dir
        self.inference_mode = inference_mode
        self.swag_training_epochs = swag_training_epochs
        self.swag_lr = swag_lr
        self.swag_update_interval = swag_update_interval
        self.max_rank_deviation_matrix = max_rank_deviation_matrix
        self.num_bma_samples = num_bma_samples

        # Network used to perform SWAG.
        # Note that all operations in this class modify this network IN-PLACE!
        self.network = CNN(in_channels=3, out_classes=6)

        # Store training dataset to recalculate batch normalization statistics during SWAG inference
        self.training_dataset = torch.utils.data.TensorDataset(train_xs)

        # SWAG-diagonal and SWAG-full
        self.n_models_collected = 0
        self.mean_weights = self._create_weight_copy()
        self.square_mean_weights = self._create_weight_copy()

        # SWAG-full attributes
        if self.inference_mode == InferenceType.SWAG_FULL:
            self.deviation_matrices = {
                name: []
                for name, param in self.network.named_parameters()
            }

        # Calibration attributes
        self.mi_threshold = None  # For OOD detection based on mutual information

    def update_swag_statistics(self) -> None:
        """
        Update SWAG statistics with the current weights of self.network.
        """
        # Create a copy of the current network weights
        copied_params = {name: param.detach().clone() for name, param in self.network.named_parameters()}

        # Update mean and square of parameters
        for name, param in copied_params.items():
            if self.n_models_collected == 0:
                self.mean_weights[name] = param.clone()
                self.square_mean_weights[name] = param.clone() ** 2
            else:
                # Update running mean
                self.mean_weights[name] += (param - self.mean_weights[name]) / (self.n_models_collected + 1)
                # Update running mean of squares
                self.square_mean_weights[name] += (param ** 2 - self.square_mean_weights[name]) / (self.n_models_collected + 1)

        # SWAG-full: Store deviations centered around the mean
        if self.inference_mode == InferenceType.SWAG_FULL:
            for name, param in copied_params.items():
                deviation = (param - self.mean_weights[name]).clone()
                if len(self.deviation_matrices[name]) >= self.max_rank_deviation_matrix:
                    self.deviation_matrices[name].pop(0)
                self.deviation_matrices[name].append(deviation)

        self.n_models_collected += 1

    def fit_swag_model(self, loader: torch.utils.data.DataLoader) -> None:
        """
        Fit SWAG on top of the pretrained network self.network.
        This method should perform gradient descent with occasional SWAG updates
        by calling self.update_swag_statistics().
        """

        # We use SGD with momentum and weight decay to perform SWA.
        optimizer = torch.optim.SGD(
            self.network.parameters(),
            lr=self.swag_lr,
            momentum=0.9,
            nesterov=False,
            weight_decay=1e-4,
        )
        loss_fn = torch.nn.CrossEntropyLoss(
            reduction="mean",
        )
        # Cosine annealing learning rate scheduler
        lr_scheduler = SWAGScheduler(
            optimizer,
            epochs=self.swag_training_epochs,
            steps_per_epoch=len(loader),
            swag_lr=self.swag_lr,
        )

        self.n_models_collected = 0

        self.network.train()
        with tqdm.trange(self.swag_training_epochs, desc="Running gradient descent for SWAG") as pbar:
            progress_dict = {}
            for epoch in pbar:
                avg_loss = 0.0
                avg_accuracy = 0.0
                num_samples = 0
                for batch_images, batch_snow_labels, batch_cloud_labels, batch_labels in loader:
                    optimizer.zero_grad()
                    predictions = self.network(batch_images)
                    batch_loss = loss_fn(input=predictions, target=batch_labels)
                    batch_loss.backward()
                    optimizer.step()
                    progress_dict["lr"] = lr_scheduler.get_last_lr()[0]
                    lr_scheduler.step()

                    # Calculate cumulative average training loss and accuracy
                    avg_loss = (batch_images.size(0) * batch_loss.item() + num_samples * avg_loss) / (
                        num_samples + batch_images.size(0)
                    )
                    avg_accuracy = (
                        torch.sum(predictions.argmax(dim=-1) == batch_labels).item()
                        + num_samples * avg_accuracy
                    ) / (num_samples + batch_images.size(0))
                    num_samples += batch_images.size(0)
                    progress_dict["avg. epoch loss"] = avg_loss
                    progress_dict["avg. epoch accuracy"] = avg_accuracy
                    pbar.set_postfix(progress_dict)

                # Periodically update SWAG statistics
                if (epoch + 1) % self.swag_update_interval == 0:
                    self.update_swag_statistics()

    def apply_calibration(self, validation_data: torch.utils.data.Dataset) -> None:
        """
        Calibrate the model using mutual information on the validation set.
        """
        val_images, _, _, val_labels = validation_data.tensors
        non_ambiguous_mask = val_labels != -1
        val_labels_non_ambiguous = val_labels[non_ambiguous_mask]
        val_images_non_ambiguous = val_images[non_ambiguous_mask]

        # Get list of probabilities from SWAG samples
        probabilities_list = self.collect_probabilities(val_images_non_ambiguous)
        # Compute mutual information
        mutual_info = self.compute_mutual_information(probabilities_list)

        # Determine the optimal MI threshold
        threshold_values = torch.linspace(mutual_info.min(), mutual_info.max(), steps=100)
        costs = []
        for threshold in threshold_values:
            # Predict "don't know" for samples with MI above the threshold
            mean_probs = torch.mean(torch.stack(probabilities_list), dim=0)
            predictions = torch.where(
                mutual_info <= threshold,
                torch.argmax(mean_probs, dim=-1),
                -1 * torch.ones_like(val_labels_non_ambiguous)
            )
            cost = cost_function(predictions, val_labels_non_ambiguous).item()
            costs.append(cost)

        best_threshold_index = np.argmin(costs)
        self.mi_threshold = threshold_values[best_threshold_index].item()
        print(f"Optimal MI threshold: {self.mi_threshold:.4f}")

    def compute_mutual_information(self, probabilities_list: typing.List[torch.Tensor]) -> torch.Tensor:
        """
        Compute mutual information across model samples.
        """
        # Stack probabilities from different samples: [num_samples, N, C]
        probs_stack = torch.stack(probabilities_list, dim=0)
        mean_probs = torch.mean(probs_stack, dim=0)  # [N, C]
        entropy_mean = -torch.sum(mean_probs * torch.log(mean_probs + 1e-12), dim=-1)  # [N]
        mean_entropy = -torch.mean(torch.sum(probs_stack * torch.log(probs_stack + 1e-12), dim=-1), dim=0)  # [N]
        mutual_info = entropy_mean - mean_entropy
        return mutual_info

    def sample_parameters(self) -> None:
        """
        Sample a new network from the approximate SWAG posterior.
        For simplicity, this method directly modifies self.network in-place.
        Hence, after calling this method, self.network corresponds to a new posterior sample.
        """

        # Instead of acting on a full vector of parameters, all operations can be done on per-layer parameters.
        for name, param in self.network.named_parameters():
            mean_weights = self.mean_weights[name]
            variance_weights = self.square_mean_weights[name] - mean_weights ** 2
            std_weights = torch.sqrt(variance_weights.clamp(min=1e-30))

            # Diagonal sampling
            z_diag = torch.randn_like(param)
            sampled_weight = mean_weights + std_weights * z_diag

            # Full SWAG sampling
            if self.inference_mode == InferenceType.SWAG_FULL and len(self.deviation_matrices[name]) >= 2:
                K = len(self.deviation_matrices[name])
                deviation_matrix = torch.stack(self.deviation_matrices[name], dim=-1)  # Shape: param_size x K
                z_full = torch.randn(K)
                scaled_deviation = (deviation_matrix @ z_full) / math.sqrt(K - 1)
                sampled_weight += scaled_deviation

            # Modify weight value in-place
            param.data.copy_(sampled_weight)

        # Update batch normalization statistics
        self._update_batchnorm_statistics()

    def predict_labels(self, predicted_probabilities: torch.Tensor) -> torch.Tensor:
        """
        Predict labels using mutual information and the calibrated MI threshold.
        """
        # Since in our updated implementation, predicted_probabilities is a list of tensors
        probabilities_list = predicted_probabilities
        mean_probs = torch.mean(torch.stack(probabilities_list), dim=0)
        mutual_info = self.compute_mutual_information(probabilities_list)
        max_likelihood_labels = torch.argmax(mean_probs, dim=-1)

        # Predict "don't know" for samples with high mutual information
        return torch.where(
            mutual_info <= self.mi_threshold,
            max_likelihood_labels,
            -1 * torch.ones_like(max_likelihood_labels)
        )

    def _create_weight_copy(self) -> typing.Dict[str, torch.Tensor]:
        """Create an all-zero copy of the network weights as a dictionary that maps name -> weight"""
        return {
            name: torch.zeros_like(param, requires_grad=False)
            for name, param in self.network.named_parameters()
        }

    def fit(
        self,
        loader: torch.utils.data.DataLoader,
    ) -> None:
        """
        Perform full SWAG fitting procedure.
        If `PRETRAINED_WEIGHTS_FILE` is `True`, this method skips the MAP inference part,
        and uses pretrained weights instead.

        Note that MAP inference can take a long time.
        You should hence only perform MAP inference yourself after passing the hard baseline
        using the given CNN architecture and pretrained weights.
        """

        # MAP inference to obtain initial weights
        PRETRAINED_WEIGHTS_FILE = self.model_dir / "map_weights.pt"
        if USE_PRETRAINED_MODEL:
            self.network.load_state_dict(torch.load(PRETRAINED_WEIGHTS_FILE))
            print("Loaded pretrained MAP weights from", PRETRAINED_WEIGHTS_FILE)
        else:
            self.fit_map_model(loader)

        # SWAG
        if self.inference_mode in (InferenceType.SWAG_DIAGONAL, InferenceType.SWAG_FULL):
            self.fit_swag_model(loader)

    def fit_map_model(self, loader: torch.utils.data.DataLoader) -> None:
        """
        MAP inference procedure to obtain initial weights of self.network.
        This is the exact procedure that was used to obtain the pretrained weights we provide.
        """
        map_training_epochs = 140
        initial_learning_rate = 0.01
        reduced_learning_rate = 0.0001
        start_decay_epoch = 50
        decay_factor = reduced_learning_rate / initial_learning_rate

        # Create optimizer, loss, and a learning rate scheduler that aids convergence
        optimizer = torch.optim.SGD(
            self.network.parameters(),
            lr=initial_learning_rate,
            momentum=0.9,
            nesterov=False,
            weight_decay=1e-4,
        )
        loss_fn = torch.nn.CrossEntropyLoss(
            reduction="mean",
        )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            [
                torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0),
                torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=1.0,
                    end_factor=decay_factor,
                    total_iters=(map_training_epochs - start_decay_epoch) * len(loader),
                ),
            ],
            milestones=[start_decay_epoch * len(loader)],
        )

        # Put network into training mode
        self.network.train()
        with tqdm.trange(map_training_epochs, desc="Fitting initial MAP weights") as pbar:
            progress_dict = {}
            # Perform the specified number of MAP epochs
            for epoch in pbar:
                avg_loss = 0.0
                avg_accuracy = 0.0
                num_samples = 0
                # Iterate over batches of randomly shuffled training data
                for batch_images, _, _, batch_labels in loader:
                    # Training step
                    optimizer.zero_grad()
                    predictions = self.network(batch_images)
                    batch_loss = loss_fn(input=predictions, target=batch_labels)
                    batch_loss.backward()
                    optimizer.step()

                    # Save learning rate that was used for step, and calculate new one
                    progress_dict["lr"] = lr_scheduler.get_last_lr()[0]
                    with warnings.catch_warnings():
                        # Suppress annoying warning (that we cannot control) inside PyTorch
                        warnings.simplefilter("ignore")
                        lr_scheduler.step()

                    # Calculate cumulative average training loss and accuracy
                    avg_loss = (batch_images.size(0) * batch_loss.item() + num_samples * avg_loss) / (
                        num_samples + batch_images.size(0)
                    )
                    avg_accuracy = (
                        torch.sum(predictions.argmax(dim=-1) == batch_labels).item()
                        + num_samples * avg_accuracy
                    ) / (num_samples + batch_images.size(0))
                    num_samples += batch_images.size(0)

                    progress_dict["avg. epoch loss"] = avg_loss
                    progress_dict["avg. epoch accuracy"] = avg_accuracy
                    pbar.set_postfix(progress_dict)

    def predict_probabilities(self, xs: torch.Tensor) -> torch.Tensor:
        """
        Predict class probabilities for the given images xs.
        This method returns a list of tensors,
        where each tensor corresponds to the predicted probabilities from one SWAG sample.
        """
        self.network = self.network.eval()

        if self.inference_mode == InferenceType.MAP:
            # For MAP, we only need one prediction
            loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(xs),
                batch_size=32,
                shuffle=False,
                num_workers=0,
                drop_last=False,
            )
            with torch.no_grad():
                return [self.predict_probabilities_map(loader)]
        else:
            # For SWAG, collect probabilities from multiple samples
            return self.collect_probabilities(xs)

    def collect_probabilities(self, xs: torch.Tensor) -> typing.List[torch.Tensor]:
        """
        Collect probabilities from SWAG samples.
        """
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(xs),
            batch_size=32,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )
        probabilities_list = []
        self.network.eval()
        for _ in tqdm.trange(self.num_bma_samples, desc="Performing Bayesian model averaging"):
            self.sample_parameters()
            probs = []
            for (batch_images,) in loader:
                with torch.no_grad():
                    outputs = self.network(batch_images)
                    probs.append(torch.softmax(outputs, dim=-1))
            probabilities_list.append(torch.cat(probs, dim=0))
        return probabilities_list

    def predict_probabilities_map(self, loader: torch.utils.data.DataLoader) -> torch.Tensor:
        """
        Predict probabilities assuming that self.network is a MAP estimate.
        This simply performs a forward pass for every batch in `loader`,
        concatenates all results, and applies a row-wise softmax.
        """
        all_predictions = []
        for (batch_images,) in loader:
            outputs = self.network(batch_images)
            all_predictions.append(outputs)

        all_predictions = torch.cat(all_predictions)
        return torch.softmax(all_predictions, dim=-1)

    def _update_batchnorm_statistics(self) -> None:
        """
        Reset and fit batch normalization statistics using the training dataset self.training_dataset.
        We provide this method for you for convenience.
        See the SWAG paper for why this is required.

        Batch normalization usually uses an exponential moving average, controlled by the `momentum` parameter.
        However, we are not training but want the statistics for the full training dataset.
        Hence, setting `momentum` to `None` tracks a cumulative average instead.
        The following code stores original `momentum` values, sets all to `None`,
        and restores the previous hyperparameters after updating batchnorm statistics.
        """

        original_momentum_values = dict()
        for module in self.network.modules():
            # Only need to handle batchnorm modules
            if not isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                continue

            # Store old momentum value before removing it
            original_momentum_values[module] = module.momentum
            module.momentum = None

            # Reset batch normalization statistics
            module.reset_running_stats()

        loader = torch.utils.data.DataLoader(
            self.training_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )

        self.network.train()
        for (batch_images,) in loader:
            self.network(batch_images)
        self.network.eval()

        # Restore old `momentum` hyperparameter values
        for module, momentum in original_momentum_values.items():
            module.momentum = momentum


class SWAGScheduler(torch.optim.lr_scheduler.LRScheduler):
    """
    Custom learning rate scheduler that calculates a different learning rate each gradient descent step.
    Uses a cosine annealing schedule.
    """

    def calculate_lr(self, current_epoch: float, previous_lr: float) -> float:
        """
        Calculate the learning rate for the epoch given by current_epoch.
        current_epoch is the fractional epoch of SWA fitting, starting at 0.
        That is, an integer value x indicates the start of epoch (x+1),
        and non-integer values x.y correspond to steps in between epochs (x+1) and (x+2).
        previous_lr is the previous learning rate.

        This method returns the new learning rate according to cosine annealing.
        """
        # Cosine annealing schedule
        current_step = self.last_epoch
        lr = self.initial_lr * 0.5 * (1 + math.cos(math.pi * current_step / self.total_steps))
        return lr

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        epochs: int,
        steps_per_epoch: int,
        swag_lr: float,
    ):
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.total_steps = epochs * steps_per_epoch
        self.initial_lr = optimizer.param_groups[0]['lr']
        self.swag_lr = swag_lr
        super().__init__(optimizer, last_epoch=-1, verbose=False)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, please use `get_last_lr()`.", UserWarning
            )
        return [
            self.calculate_lr(self.last_epoch / self.steps_per_epoch, group["lr"])
            for group in self.optimizer.param_groups
        ]


def evaluate(
    swag_inference: SWAGInference,
    eval_dataset: torch.utils.data.Dataset,
    extended_evaluation: bool,
    output_location: pathlib.Path,
) -> None:
    """
    Evaluate your model.
    Feel free to change or extend this code.
    :param swag_inference: Trained model to evaluate
    :param eval_dataset: Validation dataset
    :param: extended_evaluation: If True, generates additional plots
    :param output_location: Directory into which extended evaluation plots are saved
    """

    print("Evaluating model on validation data")

    # We ignore is_snow and is_cloud here, but feel free to use them as well
    images, snow_labels, cloud_labels, labels = eval_dataset.tensors

    # Predict class probabilities on test data,
    # most likely classes (according to the max predicted probability),
    # and classes as predicted by your SWAG implementation.
    probabilities_list = swag_inference.predict_probabilities(images)
    mean_probabilities = torch.mean(torch.stack(probabilities_list), dim=0)
    max_pred_probabilities, argmax_pred_labels = torch.max(mean_probabilities, dim=-1)
    predicted_labels = swag_inference.predict_labels(probabilities_list)

    # Create a mask that ignores ambiguous samples (those with class -1)
    non_ambiguous_mask = labels != -1

    # Calculate three kinds of accuracy:
    # 1. Overall accuracy, counting "don't know" (-1) as its own class
    # 2. Accuracy on all samples that have a known label. Predicting -1 on those counts as wrong here.
    # 3. Accuracy on all samples that have a known label w.r.t. the class with the highest predicted probability.
    overall_accuracy = torch.mean((predicted_labels == labels).float()).item()
    non_ambiguous_accuracy = torch.mean((predicted_labels[non_ambiguous_mask] == labels[non_ambiguous_mask]).float()).item()
    non_ambiguous_argmax_accuracy = torch.mean(
        (argmax_pred_labels[non_ambiguous_mask] == labels[non_ambiguous_mask]).float()
    ).item()
    print(f"Accuracy (raw): {overall_accuracy:.4f}")
    print(f"Accuracy (non-ambiguous only, your predictions): {non_ambiguous_accuracy:.4f}")
    print(f"Accuracy (non-ambiguous only, predicting most-likely class): {non_ambiguous_argmax_accuracy:.4f}")

    # Determine which threshold would yield the smallest cost on the validation data
    mutual_info = swag_inference.compute_mutual_information(probabilities_list)
    threshold_values = torch.linspace(mutual_info.min(), mutual_info.max(), steps=100)
    costs = []
    for threshold in threshold_values:
        predictions = torch.where(
            mutual_info <= threshold,
            argmax_pred_labels,
            -1 * torch.ones_like(argmax_pred_labels)
        )
        costs.append(cost_function(predictions, labels).item())
    best_threshold_index = np.argmin(costs)
    print(f"Best cost {costs[best_threshold_index]} at threshold {threshold_values[best_threshold_index]}")
    print("Note that this threshold does not necessarily generalize to the test set!")

    # Calculate ECE and plot the calibration curve
    calibration_data = calc_calibration_curve(mean_probabilities.numpy(), labels.numpy(), num_bins=20)
    print("Validation ECE:", calibration_data["ece"])

    if extended_evaluation:
        print("Plotting reliability diagram")
        fig = draw_reliability_diagram(calibration_data)
        fig.savefig(output_location / "reliability_diagram.pdf")

        sorted_confidence_indices = torch.argsort(max_pred_probabilities)

        # Plot samples your model is most confident about
        print("Plotting most confident validation set predictions")
        most_confident_indices = sorted_confidence_indices[-10:]
        fig, ax = plt.subplots(4, 5, figsize=(13, 11))
        for row in range(0, 4, 2):
            for col in range(5):
                sample_index = most_confident_indices[5 * row // 2 + col]
                ax[row, col].imshow(images[sample_index].permute(1, 2, 0).numpy())
                ax[row, col].set_axis_off()
                ax[row + 1, col].set_title(f"pred. {predicted_labels[sample_index]}, true {labels[sample_index]}")
                bar_colors = ["C0"] * 6
                if labels[sample_index] >= 0:
                    bar_colors[labels[sample_index]] = "C1"
                ax[row + 1, col].bar(
                    np.arange(6), mean_probabilities[sample_index].numpy(), tick_label=np.arange(6), color=bar_colors
                )
        fig.suptitle("Most confident predictions", size=20)
        fig.savefig(output_location / "examples_most_confident.pdf")

        # Plot samples your model is least confident about
        print("Plotting least confident validation set predictions")
        least_confident_indices = sorted_confidence_indices[:10]
        fig, ax = plt.subplots(4, 5, figsize=(13, 11))
        for row in range(0, 4, 2):
            for col in range(5):
                sample_index = least_confident_indices[5 * row // 2 + col]
                ax[row, col].imshow(images[sample_index].permute(1, 2, 0).numpy())
                ax[row, col].set_axis_off()
                ax[row + 1, col].set_title(f"pred. {predicted_labels[sample_index]}, true {labels[sample_index]}")
                bar_colors = ["C0"] * 6
                if labels[sample_index] >= 0:
                    bar_colors[labels[sample_index]] = "C1"
                ax[row + 1, col].bar(
                    np.arange(6), mean_probabilities[sample_index].numpy(), tick_label=np.arange(6), color=bar_colors
                )
        fig.suptitle("Least confident predictions", size=20)
        fig.savefig(output_location / "examples_least_confident.pdf")


class CNN(torch.nn.Module):
    """
    Small convolutional neural network used in this task.
    You should not modify this class before passing the hard baseline.

    Note that if you change the architecture of this network,
    you need to re-run MAP inference and cannot use the provided pretrained weights anymore.
    Hence, you need to set `USE_PRETRAINED_INIT = False` at the top of this file.
    """
    def __init__(
        self,
        in_channels: int,
        out_classes: int,
    ):
        super().__init__()

        self.layer0 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 32, kernel_size=5),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
        )
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 32, kernel_size=3),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 32, kernel_size=3),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
        )
        self.pool1 = torch.nn.MaxPool2d((2, 2), stride=(2, 2))

        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
        )
        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, kernel_size=3),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
        )
        self.pool2 = torch.nn.MaxPool2d((2, 2), stride=(2, 2))

        self.layer5 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, kernel_size=3),
        )

        self.global_pool = torch.nn.AdaptiveAvgPool2d((1, 1))

        self.linear = torch.nn.Linear(64, out_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.pool1(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool2(x)
        x = self.layer5(x)

        # Average features over both spatial dimensions, and remove the now superfluous dimensions
        x = self.global_pool(x).squeeze(-1).squeeze(-1)

        logits = self.linear(x)

        return logits


if __name__ == "__main__":
    main()
