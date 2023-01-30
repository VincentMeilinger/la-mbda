import train_utils as train_utils
from plot import make_statistics, parse_experiment_data
import os

# Bayesian optimization imports
import torch
from torch import tensor
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.utils import standardize
import botorch.acquisition as acqf
from botorch.optim import optimize_acqf


def get_y_sample(root="results", algo="la_mbda", environment="point_goal2"):
    experiment = os.path.join(root, algo, environment)
    experiment_statistics = make_statistics(*parse_experiment_data(
        experiment,
        int(1e6)))
    arr = experiment_statistics['objectives_median'][-1]  # 'Average reward return'
    acr = experiment_statistics['mean_sum_costs_median'][-1]  # 'Average cost return'
    cr = experiment_statistics['average_costs_median'][-1]  # 'Cost regret'
    return torch.tensor([[cr + acr]])


if __name__ == '__main__':
    config_dict = train_utils.define_config()
    config_dict["log_dir"] = 'results/la_mbda/point_goal2/314'
    config_dict["safety"] = True
    config = train_utils.make_config(config_dict)
    from la_mbda.la_mbda import LAMBDA

    bo_vars = [
        {
            "name": "discount",
            "bounds": tensor([0.9, 1.0, 0.01])
        },
        {
            "name": "lambda_",
            "bounds": tensor([0.9, 1.0, 0.01])
        },
    ]  # 'discount': 0.99, 'lambda_': 0.95

    bounds = tensor([[0., 0.]])
    for var in bo_vars:
        bound = torch.unsqueeze(var["bounds"][0:2], 0)
        bounds = torch.cat((bounds, bound), dim=0)
    bounds = bounds[1:].T
    print("*** Bounds: ", bounds)

    print("*** Starting bayesian optimization of hyper parameters ", [var["name"] + ", " for var in bo_vars])
    print("*** Computing warmup sample ...")

    # Compute initial sample
    X_samples = tensor([[torch.rand(1), torch.rand(1)]])
    print("*** Warmup candidate: ", X_samples[0])
    train_utils.train(config, LAMBDA)
    y_samples = get_y_sample(root="results/")
    print("*** Sample value: ", float(y_samples[0]))

    n_iters = 3
    for i in range(n_iters):
        print("\n\n*** Iteration ", i)

        # Create config, set values
        config_dict = train_utils.define_config()
        data_path_root = "results" + str(i) + "/"
        config_dict["data_path"] = data_path_root
        config_dict["log_dir"] = data_path_root + "la_mbda/point_goal2/314"
        config_dict["safety"] = True
        print("*** Log directory: ", config_dict["log_dir"])

        # Fit samples to gp
        gp = SingleTaskGP(X_samples, y_samples)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_model(mll)

        # Define and sample the acquisition function
        EI = acqf.ExpectedImprovement(
            gp,
            y_samples.min(),
            maximize=False
        )

        best_candidate, acq_value = optimize_acqf(
            acq_function=EI,
            bounds=bounds,
            q=1,
            num_restarts=1,
            raw_samples=20
        )

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




