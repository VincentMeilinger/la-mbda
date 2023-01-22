import train_utils as train_utils

# Bayesian optimization imports
import torch
from torch import tensor
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.utils import standardize
from gpytorch.mlls import ExactMarginalLogLikelihood

import botorch.acquisition as acqf
from botorch.optim import optimize_acqf

if __name__ == '__main__':
    config = train_utils.make_config(train_utils.define_config())
    from la_mbda.la_mbda import LAMBDA

    bo_vars = {
        'discount': (0.9, 1.0, 0.01),
        'lambda_': (0.9, 1.0, 0.01)
    }  # 'discount': 0.99, 'lambda_': 0.95

    bounds = []
    for val in bo_vars.values():
        bounds.append(tensor(val))
    X_samples = tensor([[torch.rand((1)), torch.rand((1))]])
    y_samples = sample(X_samples[:, 0], X_samples[:, 1])[0].reshape(-1, 1)
    print("=======================")
    print("initialization")
    print("candidate: ", X_samples[0])
    print("model score for candidate: ", float(y_samples[0]))
    n_iters = 1
    for i in range(n_iters):
        print("\n\n*** Iteration: ", i)
        # print(X_samples)
        gp = SingleTaskGP(X_samples, y_samples)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_model(mll)

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

        y_new_sample, K = sample(best_candidate[0][0],
                                best_candidate[0][1])
        # print("candidate: ", best_candidate)
        # print("model score for candidate: ", float(y_new_sample))

        X_samples = torch.vstack((X_samples, best_candidate))
        y_samples = torch.vstack((y_samples, y_new_sample))



    train_utils.train(config, LAMBDA)
