import torch
import numpy as np
import math

def _flatten(sequence):
    flat = [p.contiguous().view(-1) for p in sequence]
    return torch.cat(flat) if len(flat) > 0 else torch.tensor([])


def compute_cc_weights(nb_steps):
    lam = np.arange(0, nb_steps + 1, 1).reshape(-1, 1)
    lam = np.cos((lam @ lam.T) * math.pi / nb_steps) #(nb_steps + 1, nb_steps + 1)
    lam[:, 0] = .5
    lam[:, -1] = .5 * lam[:, -1]
    lam = lam * 2 / nb_steps
    W = np.arange(0, nb_steps + 1, 1).reshape(-1, 1) #(nb_steps + 1, 1)
    W[np.arange(1, nb_steps + 1, 2)] = 0
    W = 2 / (1 - W ** 2)
    W[0] = 1
    W[np.arange(1, nb_steps + 1, 2)] = 0
    cc_weights = torch.tensor(lam.T @ W).float() #(nb_steps +1, 1)
    steps = torch.tensor(np.cos(np.arange(0, nb_steps + 1, 1).reshape(-1, 1) * math.pi / nb_steps)).float() #(nb_steps +1, 1)

    return cc_weights, steps


def integrate(x0, nb_steps, step_sizes, integrand, compute_grad=False, x_tot=None):
    #Clenshaw-Curtis Quadrature Method
    #x_tot is gradient of Loss (a scalar) wrt integrate() which has output dim = input dim across each channel
    #Thus shape of x_tot: (batch_size, num channels, input_dim)
    cc_weights, steps = compute_cc_weights(nb_steps) # both shape: (nb_steps +1, 1)

    device = x0.get_device() if x0.is_cuda  else "cpu"

    cc_weights, steps = cc_weights.to(device), steps.to(device) # both shape: (nb_steps +1, 1)

    xT = x0 + nb_steps*step_sizes # (batch_size, num channels, input_dim)
    batch_size, num_channels, in_dim = x0.shape
    if not compute_grad:
        x0_t = x0.unsqueeze(1).expand(-1, nb_steps+1, -1, -1) # (batch_size, nb_steps+1, num channels, input_dim), nb_steps+1 identical copies
        xT_t = xT.unsqueeze(1).expand(-1, nb_steps+1, -1, -1) # (batch_size, nb_steps+1, num channels, input_dim), nb_steps+1 identical copies
        steps_t = steps.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, num_channels, in_dim) # (batch_size, nb_steps+1, num channels, input_dim), copies of dim1 across dim0, dim2, dim3
        X_steps = x0_t + (xT_t-x0_t)*(steps_t + 1)/2 # (batch_size, nb_steps+1, num channels, input_dim), X_i for step_i
        X_steps = X_steps.contiguous().view(-1, num_channels, in_dim) # (batch_size*(nb_steps+1), num channels, input_dim) : [everything in batch_1, everything in batch_2, ...]
        dzs = integrand(X_steps) #(batch_size*(nb_steps+1), num channels, input_dim)
        dzs = dzs.view(xT_t.shape[0], nb_steps+1, num_channels, -1) #back to (batch size, nb_steps + 1, input_dim)
        dzs = dzs*cc_weights.unsqueeze(0).unsqueeze(-1).expand(dzs.shape) # (nb_steps +1, 1) -> (1, nb_steps +1, 1, 1) -> (batch_size, nb_steps+1, num channels, input_dim)
        z_est = dzs.sum(1) # (batch_size, num channels, input dim)
        return z_est*(xT - x0)/2 # (batch_size, num channels, input dim) if input dim and output dim are same, they have to be
    else:

        x0_t = x0.unsqueeze(1).expand(-1, nb_steps+1, -1, -1) # (batch_size, nb_steps+1, num channels, input_dim), nb_steps+1 identical copies
        xT_t = xT.unsqueeze(1).expand(-1, nb_steps+1, -1, -1) # (batch_size, nb_steps+1, num channels, input_dim), nb_steps+1 identical copies
        x_tot = x_tot * (xT - x0) / 2 # (batch_size, num channels, input_dim)
        x_tot_steps = x_tot.unsqueeze(1).expand(-1, nb_steps + 1, -1, -1) * cc_weights.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, num_channels, in_dim) # (batch_size, nb_steps+1, num channels, input_dim)
        steps_t = steps.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, num_channels, in_dim) # (batch_size, nb_steps+1, num channels, input_dim), copies of dim1 across dim0, dim2, dim3
        X_steps = x0_t + (xT_t - x0_t) * (steps_t + 1) / 2 # (batch_size, nb_steps+1, num channels, input_dim)
        X_steps = X_steps.contiguous().view(-1, num_channels, in_dim) # (batch_size*(nb_steps+1), num channels, input_dim) : [everything in batch_1, everything in batch_2, ...]
        x_tot_steps = x_tot_steps.contiguous().view(-1, num_channels, in_dim) # (batch_size*(nb_steps+1), num channels, input_dim) : [everything in batch_1, everything in batch_2, ...]

        g_param = computeIntegrand(X_steps, integrand, x_tot_steps) #(num_model_params,)
        return g_param


def computeIntegrand(x, integrand, x_tot):
    with torch.enable_grad():
        f = integrand.forward(x)
        g_param = _flatten(torch.autograd.grad(f, integrand.parameters(), x_tot, create_graph=True, retain_graph=True)) #(num_model_params,)

    return g_param


class ParallelNeuralIntegral(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x0, x, integrand, flat_params, nb_steps=20):
        with torch.no_grad():
            #x: (bsize, channels, in dim)
            x_tot = integrate(x0, nb_steps, (x - x0)/nb_steps, integrand, False)
            # Save for backward
            ctx.integrand = integrand
            ctx.nb_steps = nb_steps
            ctx.save_for_backward(x0.clone(), x.clone())
        return x_tot

    @staticmethod
    def backward(ctx, grad_output):
        #grad output of shape (bsize, num channels, in dim)
        x0, x = ctx.saved_tensors
        integrand = ctx.integrand
        nb_steps = ctx.nb_steps
        integrand_grad = integrate(x0, nb_steps, (x-x0)/nb_steps, integrand, True, grad_output)
        x_grad = integrand(x)
        x0_grad = integrand(x0)
        # Leibniz formula
        return -x0_grad*grad_output, x_grad*grad_output, None, integrand_grad, None