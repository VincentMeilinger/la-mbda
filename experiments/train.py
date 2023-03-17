import train_utils as train_utils
from plot import make_statistics, parse_experiment_data
import os

# Bayesian optimization imports
import torch
from torch import tensor
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
import botorch.acquisition as acqf
from botorch.optim import optimize_acqf
import matplotlib.pyplot as plt


def plot_bo(gp, ei, x_samples, iter, bounds):
    """
    Create and save plots of the Gaussian process posterior mean and the acquisition function within the BO loop.
    Plots are saved in the "plots/" directory.
    Arguments:
        gp: Gaussian Process
        ei: "Expected Improvement" acquisition function
        x_samples: All previous observations of the objective function
        iter: Current BO iteration
        bounds: Bounds of the variables to optimize
    """

    x_ = torch.linspace(-1, 1, 100)
    y_ = torch.linspace(1, -1, 100)
    x_axis, y_axis = torch.meshgrid(x_, y_, indexing="xy")
    grid = torch.stack((x_axis, y_axis), 2)
    X_ = grid.reshape(len(x_) * len(y_), 2)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.title.set_text('Acquisition Function')

    if len(x_samples) > 1:
        ax.scatter(x_samples[:, 0][0:-1], x_samples[:, 1][0:-1], s=35, color='white', edgecolors='black')
    ax.scatter(x_samples[:, 0][-1], x_samples[:, 1][-1], s=35, color='red', edgecolors='black')

    ei_post = ei(X_.unsqueeze(1)).detach().numpy().reshape(len(x_), len(y_))
    ax.imshow(ei_post, extent=[bounds[0][0], bounds[1][0], bounds[0][1], bounds[1][1]])
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_xlabel('discount')
    ax.set_ylabel('lambda_')
    plt.savefig("plots/acqf" + str(iter) + ".png")

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.title.set_text('GP Mean')

    if len(x_samples) > 1:
        ax.scatter(x_samples[:, 0][0:-1].numpy(), x_samples[:, 1][0:-1], s=35, color='white', edgecolors='black')
    ax.scatter(x_samples[:, 0][-1], x_samples[:, 1][-1], s=35, color='red', edgecolors='black')

    posterior_mean = gp.posterior(X_).mean.detach().numpy().reshape(len(x_), len(y_))
    ax.imshow(posterior_mean, extent=[bounds[0][0], bounds[1][0], bounds[0][1], bounds[1][1]])

    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_xlabel('discount')
    ax.set_ylabel('lambda_')
    plt.savefig("plots/mean" + str(iter) + ".png")


def plot_scores(y_samples):
    """
    Create and save a plot summarizing the cost in each BO iteration.
    Arguments:
        y_samples: Tensor of cost function observations of each iteration
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.title.set_text('Agent score over iterations')
    ax.plot(y_samples, color='green', marker='o', linestyle='dashed', linewidth=2, markersize=12)
    plt.savefig("plots/score.png")


def write_to_csv(row, mode='a'):
    import csv
    # open the file in the write mode
    with open('plots/bo_samples.csv', mode) as f:
        # create the csv writer
        writer = csv.writer(f)

        # write a row to the csv file
        writer.writerow(row)

def normalize(x: torch.tensor, bounds: torch.tensor):
    """
    Normalize tensor 'x' given bounds (minimum and maximum) to a range of [-1, 1] in each dimension.
    Arguments:
        x: Input tensor to normalize
        bounds: The minimum/maximum values for each dimension ([[min_1, min_2, ...][max_1, max_2, ...]])
    """
    x_min = bounds[0] * 10
    x_max = bounds[1] * 10
    x = x * 10
    norm = (2 * (x - x_min)) / (x_max - x_min) - 1
    return norm


def denormalize(norm: torch.tensor, bounds: torch.tensor):
    """
    Denormalize (rescale to original range) tensor 'norm' given bounds (minimum and maximum) in each dimension.
    Arguments:
        norm: Input tensor to denormalize
        bounds: The minimum/maximum values for each dimension ([[min_1, min_2, ...][max_1, max_2, ...]])
    """
    x_min = bounds[0] * 10
    x_max = bounds[1] * 10
    return (((norm + 1) * (x_max - x_min))/2 + x_min)/10


def get_y_sample(root="results", algo="la_mbda", environment="point_goal2"):
    """
    Compute the cost of the BO objective function. Assumes experiments are stored in the following folder structure:
    'root/algo/environment/...'
    :param root: Root experiment directory
    :param algo: Algorithm directory
    :param environment: Environment directory
    :return: Cost function value
    """
    experiment = os.path.join(root, algo, environment)
    experiment_statistics = make_statistics(*parse_experiment_data(
        experiment,
        int(1e6)))
    arr = experiment_statistics['objectives_median'][-1]  # 'Average reward return'
    acr = experiment_statistics['mean_sum_costs_median'][-1]  # 'Average cost return'
    cr = experiment_statistics['average_costs_median'][-1]  # 'Cost regret'
    return torch.tensor([[cr * 100 + acr - arr]])


if __name__ == '__main__':
    """
    The Bayesian optimization (BO) procedure.
    """
    torch.manual_seed(0)
    config_dict = train_utils.define_config()
    config_dict["log_dir"] = 'results/la_mbda/point_goal2/314'
    config_dict["safety"] = True
    config = train_utils.make_config(config_dict)
    from la_mbda.la_mbda import LAMBDA

    # Define the hyperparameters to optimize (name and bounds)
    bo_vars = [
        {
            "name": "discount",
            "bounds": tensor([0.8, 1.0])
        },
        {
            "name": "lambda_",
            "bounds": tensor([0.8, 1.0])
        },
    ]  # 'discount': 0.99, 'lambda_': 0.95

    # Tensor containing hyperparameter bounds
    bounds = tensor([[0., 0.]])
    for var in bo_vars:
        bound = torch.unsqueeze(var["bounds"], 0)
        bounds = torch.cat((bounds, bound), dim=0)
    bounds = bounds[1:].T
    print("*** Bounds: ", bounds)
    bounds_norm = torch.tensor([[-1., -1.], [1., 1.]])

    print("*** Starting bayesian optimization of hyper parameters ", [var["name"] + ", " for var in bo_vars])
    print("*** Computing warmup sample ...")

    # Compute initial sample
    X_samples = tensor([[1.0, 1.0]])
    print("*** Warmup candidate: ", X_samples[0])
    train_utils.train(config, LAMBDA)
    y_samples = get_y_sample(root="results/")
    print("*** Sample value: ", float(y_samples[0]))
    write_to_csv((str(X_samples[0]), str(y_samples[0])), mode='w')
    n_iters = 30
    for i in range(n_iters):
        print("\n\n*** Iteration ", i)

        # Create config, set values
        config_dict = train_utils.define_config()
        data_path_root = "results/result" + str(i) + "/"
        config_dict["data_path"] = data_path_root
        config_dict["log_dir"] = data_path_root + "la_mbda/point_goal2/314"
        config_dict["safety"] = True
        print("*** Log directory: ", config_dict["log_dir"])

        # Fit samples to gp
        gp = SingleTaskGP(normalize(X_samples, bounds=bounds), y_samples)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_model(mll)

        # Define and sample the acquisition function
        ei = acqf.ExpectedImprovement(
            gp,
            y_samples.min(),
            maximize=False
        )

        best_candidate, acq_value = optimize_acqf(
            acq_function=ei,
            bounds=bounds_norm,
            q=1,
            num_restarts=1,
            raw_samples=60
        )

        best_candidate = denormalize(best_candidate, bounds=bounds)

        # Set new hyperparameters in config
        for ix, var in enumerate(bo_vars):
            config_dict[var["name"]] = float(best_candidate[0][ix])

        # Train and sample agent
        print("*** Next candidate: ", best_candidate)
        config = train_utils.make_config(config_dict)
        print("*** Config: ", config)
        train_utils.train(config, LAMBDA)
        print("*** Fetching sample ...")
        y_new_sample = get_y_sample(root=data_path_root)
        print("*** Sample value: ", float(y_new_sample))

        # Update new prior samples
        X_samples = torch.vstack((X_samples, best_candidate))
        y_samples = torch.vstack((y_samples, y_new_sample))

        # Write samples and sample locations to csv file
        write_to_csv(
            (str(X_samples[i+1]), str(y_samples[i+1])),
            mode='a'
        )

        # Plot gp posterior and acquisition function with sample locations
        plot_bo(
            gp=gp,
            ei=ei,
            x_samples=X_samples,
            iter=i,
            bounds=bounds
        )

        # Line plot of all collected model scores
        plot_scores(y_samples)
