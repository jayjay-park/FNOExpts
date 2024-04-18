import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchdiffeq
import datetime
import numpy as np
import argparse
import json
import logging
import os
import math
from functools import reduce
import operator
from matplotlib.pyplot import *
from mpl_toolkits.mplot3d import axes3d


########################
### Dynamical System ###
########################
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def lorenz(t, u, params=[10.0,28.0,8/3]):
    """ Lorenz chaotic differential equation: du/dt = f(t, u)
    t: time T to evaluate system
    u: state vector [x, y, z] 
    return: new state vector in shape of [3]"""

    du = torch.stack([
            params[0] * (u[1] - u[0]),
            u[0] * (params[1] - u[2]) - u[1],
            (u[0] * u[1]) - (params[2] * u[2])
        ])
    return du

def rossler(t, X):
    '''Parameter values picked from: The study of Lorenz and Rössler strange attractors by means of quantum theory by Bogdanov et al.
    https://arxiv.org/ftp/arxiv/papers/1412/1412.2242.pdf
    LE:  0.07062, 0.000048, -5.3937
    '''
    x, y, z = X
    a = 0.2
    b = 0.2
    c = 5.7
    
    dx = -(y + z)
    dy = x + a * y
    dz = b + z * (x - c)
    return torch.stack([dx, dy, dz])

################################################################
#  1d fourier layer
################################################################
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  #Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        res = torch.einsum("bix,iox->box", input, weights).clone()
        return res

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes without in-place operation
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft_copy = out_ft.clone()
        # out_ft_copy[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)
        before_mode = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)
        out_ft_copy = torch.cat((before_mode, out_ft_copy[:,:,self.modes1:]), dim = 2)

        #Return to physical space
        x = torch.fft.irfft(out_ft_copy, n=x.size(-1))
        return x

class FNO1d(nn.Module):
    def __init__(self, modes, width):
        super(FNO1d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.modes1 = modes
        self.width = width
        self.padding = 2 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(2, self.width) # input channel is 2: (a(x), x)

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)
        # x = F.pad(x, [0,self.padding]) # pad the domain if input is non-periodic

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        # x = x[..., :-self.padding] # pad the domain if input is non-periodic
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)
    

# Sobolev norm (HS norm)
# where we also compare the numerical derivatives between the output and target
class HsLoss_2d(object):
    def __init__(self, d=2, p=2, k=1, a=None, group=False, size_average=True, reduction=True):
        super(HsLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.k = k
        self.balanced = group
        self.reduction = reduction
        self.size_average = size_average

        if a == None:
            a = [1,] * k
        self.a = a

    def rel(self, x, y):
        num_examples = x.size()[0]
        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)
        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)
        return diff_norms/y_norms

    def __call__(self, x, y, a=None):
        nx = x.size()[1]
        ny = x.size()[2]
        k = self.k
        balanced = self.balanced
        a = self.a
        x = x.view(x.shape[0], nx, ny, -1)
        y = y.view(y.shape[0], nx, ny, -1)

        k_x = torch.cat((torch.arange(start=0, end=nx//2, step=1),torch.arange(start=-nx//2, end=0, step=1)), 0).reshape(nx,1).repeat(1,ny)
        k_y = torch.cat((torch.arange(start=0, end=ny//2, step=1),torch.arange(start=-ny//2, end=0, step=1)), 0).reshape(1,ny).repeat(nx,1)
        k_x = torch.abs(k_x).reshape(1,nx,ny,1).to(x.device)
        k_y = torch.abs(k_y).reshape(1,nx,ny,1).to(x.device)

        x = torch.fft.fftn(x, dim=[1, 2])
        y = torch.fft.fftn(y, dim=[1, 2])

        if balanced==False:
            weight = 1
            if k >= 1:
                weight += a[0]**2 * (k_x**2 + k_y**2)
            if k >= 2:
                weight += a[1]**2 * (k_x**4 + 2*k_x**2*k_y**2 + k_y**4)
            weight = torch.sqrt(weight)
            loss = self.rel(x*weight, y*weight)
        else:
            loss = self.rel(x, y)
            if k >= 1:
                weight = a[0] * torch.sqrt(k_x**2 + k_y**2)
                loss += self.rel(x*weight, y*weight)
            if k >= 2:
                weight = a[1] * torch.sqrt(k_x**4 + 2*k_x**2*k_y**2 + k_y**4)
                loss += self.rel(x*weight, y*weight)
            loss = loss / (k+1)

        return loss


## For Lorentz System
class HsLoss(object):
    def __init__(self, d=2, p=2, k=1, a=None, size_average=True, reduction=True):
        super(HsLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.k = k
        self.reduction = reduction
        self.size_average = size_average

        if a == None:
            a = [1,] * k
        self.a = a

    def rel(self, x, y):
        num_examples = x.size()[0]
        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)
        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)
        return diff_norms/y_norms

    def __call__(self, x, y, a=None):
        nx = x.size()[1]
        ny = 1
        k = self.k
        a = self.a
        x = x.view(x.shape[0], nx, ny, -1)
        y = y.view(y.shape[0], nx, ny, -1)

        k_x = torch.tensor([[[[0.]], [[2.]], [[1.]]]], device = x.device)
        k_y = torch.ones(1,nx,ny,1, device = x.device)

        x = torch.fft.fftn(x, dim=[1, 2])
        y = torch.fft.fftn(y, dim=[1, 2])

        ## Dissipative regularization

        loss = self.rel(x, y)
        if k >= 1:
            weight = a[0] * torch.sqrt(k_x**2 + k_y**2)
            loss += self.rel(x*weight, y*weight)
        if k >= 2:
            weight = a[1] * (k_x**2 + k_y**2)
            loss += self.rel(x*weight, y*weight)
        loss = loss / (k+1)

        return loss


##############
## Training ##
##############

# print the number of parameters
def count_params(model):
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul, 
                    list(p.size()+(2,) if p.is_complex() else p.size()))
    return c

def create_data(dyn_info, n_train, n_test, n_val, n_trans):
    dyn, dim, time_step = dyn_info
    # Adjust total time to account for the validation set
    tot_time = time_step * (n_train + n_test + n_val + n_trans + 1)
    t_eval_point = torch.arange(0, tot_time, time_step)

    # Generate trajectory using the dynamical system
    traj = torchdiffeq.odeint(dyn, torch.randn(dim), t_eval_point, method='rk4', rtol=1e-8)
    traj = traj[n_trans:]  # Discard transient part

    # Create training dataset
    X_train = traj[:n_train]
    Y_train = traj[1:n_train + 1]

    # Shift trajectory for validation dataset
    traj = traj[n_train:]
    X_val = traj[:n_val]
    Y_val = traj[1:n_val + 1]

    # Shift trajectory for test dataset
    traj = traj[n_val:]
    X_test = traj[:n_test]
    Y_test = traj[1:n_test + 1]

    return [X_train, Y_train, X_val, Y_val, X_test, Y_test]

def calculate_relative_error(model, dyn, device, batch_size):
    # Simulate an orbit using the true dynamics
    time_step = 0.01  # Example timestep, adjust as needed
    orbit = torchdiffeq.odeint(dyn, torch.randn(3), torch.arange(0, 5, time_step), method='rk4', rtol=1e-8).to(device)
    
    # Compute vector field from model and true dynamics
    vf_nn = model(orbit.reshpae(batch_size, dim, 1)).detach()
    vf_true = torch.stack([dyn(0, orbit[i]) for i in range(orbit.size(0))])

    # Calculate relative error
    err = torch.linalg.norm(vf_nn - vf_true, dim=1)
    mag = torch.linalg.norm(vf_true, dim=1)
    relative_error = torch.mean(err / mag).item() * 100  # As percentage
    return relative_error

def update_lr(optimizer, epoch, total_e, origin_lr):
    """ A decay factor of 0.1 raised to the power of epoch / total_epochs. Learning rate decreases gradually as the epoch number increases towards the total number of epochs. """
    new_lr = origin_lr * (0.1 ** (epoch / float(total_e)))
    for params in optimizer.param_groups:
        params['lr'] = new_lr
    return

def train(dyn_sys_info, model, device, dataset, optim_name, criterion, epochs, lr, weight_decay, reg_param, loss_type, model_type, batch_size):

    print('cuda', torch.cuda.is_available())
    print('memory', torch.cuda.memory_allocated(), torch.cuda.max_memory_allocated())
    torch.autograd.set_detect_anomaly(True)

    # Initialize
    n_store, k  = 10, 0
    ep_num, loss_hist, test_loss_hist = torch.empty(n_store+1,dtype=int), torch.empty(n_store+1), torch.empty(n_store+1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    X_train, Y_train, X_val, Y_val, X_test, Y_test = dataset
    X_train, Y_train, X_val, Y_val, X_test, Y_test = X_train.to(device), Y_train.to(device), X_val.to(device), Y_val.to(device), X_test.to(device), Y_test.to(device)
    num_train, num_test = X_train.shape[0], X_test.shape[0]
    dyn_sys, dim, time_step = dyn_sys_info
    dyn_sys_type = "lorenz" if dyn_sys == lorenz else "rossler"
    t_eval_point = torch.linspace(0, time_step, 2).to(device)
    torch.cuda.empty_cache()
    if loss_type == "Sobolev":
        Soboloev_Loss = HsLoss()


    # Create minibtach
    x_train = X_train.reshape(num_train,dim,1)
    x_test = X_test.reshape(num_test,dim,1)
    y_train = Y_train.reshape(num_train,dim,1)
    y_test = Y_test.reshape(num_test,dim,1)
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

    # Training Loop
    min_relative_error = 1000000

    for i in range(epochs):
        model.train()
        batch_idx = 0
        full_train_loss = 0

        for x, y in train_loader:
            y_pred = model(x).cuda().view(batch_size, -1)
            y_true = y.view(batch_size, -1)
            
            optimizer.zero_grad()
            train_loss = criterion(y_pred, y_true)  * (1/time_step/time_step)
            if loss_type == "Sobolev":
                sob_loss =  Soboloev_Loss(y_pred, y_true)
                train_loss += sob_loss
            train_loss.backward()
            optimizer.step()
            full_train_loss += train_loss.item()/len(train_loader)
        update_lr(optimizer, i, epochs, args.lr)
        print(i, full_train_loss)

        # Save Training and Test History
        if i % (epochs//n_store) == 0 or (i == epochs-1):
            y_pred_test = torch.zeros(int(num_test/batch_size), batch_size, dim)
            y_true_test = torch.zeros(int(num_test/batch_size), batch_size, dim)
            with torch.no_grad():
                model.eval()
                batch_test_idx = 0

                for x, y in test_loader:
                    y_pred_test[batch_test_idx] = model(x).cuda().squeeze()
                    y_true_test[batch_test_idx] = y.squeeze()
                    batch_test_idx += 1

                # current_relative_error = calculate_relative_error(model, dyn_sys_info[0], device)
                # Check if current model has the lowest relative error so far
                test_loss = criterion(y_pred_test.view(batch_size, -1), y_true_test.view(batch_size, -1)) * (1/time_step/time_step)
                if test_loss < min_relative_error:
                    min_relative_error = test_loss
                    # Save the model
                    torch.save(model.state_dict(), f"{args.train_dir}/best_model.pth")
                    logger.info(f"Epoch {i}: New minimal relative error: {min_relative_error:.2f}%, model saved.")

                # save predicted node feature for analysis            
                logger.info("Epoch: %d Train: %.5f Test: %.5f", i, full_train_loss, test_loss.item())
                ep_num[k], loss_hist[k], test_loss_hist[k] = i, full_train_loss, test_loss.item()

                if loss_type == "Jacobian":
                    test_model_J = compute_batch_jac(0, X_test).to(device)
                    test_jac_norm_diff = criterion(Test_J, test_model_J)
                    jac_diff_train[k], jac_diff_test[k] = jac_norm_diff, test_jac_norm_diff
                    JAC_plot_path = f'{args.train_dir}JAC_'+str(i)+'.jpg'
                    # JAC_plot_path = f'../plot/Vector_field/train_{model_type}_{dyn_sys_type}/JAC_'+str(i)+'.jpg'
                    # plot_vector_field(model, path=JAC_plot_path, idx=1, t=0., N=100, device='cuda')

                k = k + 1

    if loss_type == "Jacobian":
        for i in [0, 1, 50, -2, -1]:
            print("Point:", X_train[i].detach().cpu().numpy(), "\n", "True:", "\n", True_J[i].detach().cpu().numpy(), "\n", "JAC:", "\n", cur_model_J[i].detach().cpu().numpy())
    else:
        MSE_plot_path = f'{args.train_dir}MSE_'+str(i)+'.jpg'
        # MSE_plot_path = f'../plot/Vector_field/train_{model_type}_{dyn_sys_type}/MSE_'+str(i)+'.jpg'
        # plot_vector_field(model, path=MSE_plot_path, idx=1, t=0., N=100, device='cuda')
        jac_diff_train, jac_diff_test = None, None
    # Load the best relative error model
    best_model = model
    best_model.load_state_dict(torch.load(f"{args.train_dir}/best_model.pth"))
    best_model.eval()
    RE_plot_path = f'{args.train_dir}minRE.jpg'
    # plot_vector_field(best_model, path=RE_plot_path, idx=1, t=0., N=100, device='cuda')
    return ep_num, loss_hist, test_loss_hist, jac_diff_train, jac_diff_test, Y_test



##############
#### Plot ####
##############

def plot_loss(epochs, train, test, path):
    fig, ax = subplots()
    ax.plot(epochs[30:].numpy(), train[30:].detach().cpu().numpy(), "P-", lw=2.0, ms=5.0, label="Train")
    ax.plot(epochs[30:].numpy(), test[30:].detach().cpu().numpy(), "P-", lw=2.0, ms=5.0, label="Test")
    ax.set_xlabel("Epochs",fontsize=24)
    ax.xaxis.set_tick_params(labelsize=24)
    ax.yaxis.set_tick_params(labelsize=24)
    ax.legend(fontsize=24)
    ax.grid(True)
    tight_layout()
    savefig(path, bbox_inches ='tight', pad_inches = 0.1)

def plot_attractor(model, dyn_info, time, path):
    # generate true orbit and learned orbit
    dyn, dim, time_step = dyn_info
    tran_orbit = torchdiffeq.odeint(dyn, torch.randn(dim), torch.arange(0, 5, time_step), method='rk4', rtol=1e-8)
    true_o = torchdiffeq.odeint(dyn, tran_orbit[-1], torch.arange(0, time, time_step), method='rk4', rtol=1e-8)
    # learned_o = torchdiffeq.odeint(model.eval().to(device), tran_orbit[-1].to(device), torch.arange(0, time, time_step), method="rk4", rtol=1e-8).detach().cpu().numpy()

    learned_o = torch.zeros(time*int(1/time_step), dim)
    x0 = tran_orbit[-1]
    for t in range(time*int(1/time_step)):
        learned_o[t] = x0
        new_x = model(x0.reshape(1, dim, 1).cuda())
        x0 = new_x.squeeze()
    learned_o = learned_o.detach().cpu().numpy()

    # create plot of attractor with initial point starting from 
    fig, axs = subplots(2, 3, figsize=(24,12))
    cmap = cm.plasma
    num_row, num_col = axs.shape

    for x in range(num_row):
        for y in range(num_col):
            orbit = true_o if x == 0 else learned_o
            if y == 0:
                axs[x,y].plot(orbit[0, 0], orbit[0, 1], '+', markersize=35, color=cmap.colors[0])
                axs[x,y].scatter(orbit[:, 0], orbit[:, 1], c=orbit[:, 2], s = 6, cmap='plasma', alpha=0.5)
                axs[x,y].set_xlabel("X")
                axs[x,y].set_ylabel("Y")
            elif y == 1:
                axs[x,y].plot(orbit[0, 0], orbit[0, 2], '+', markersize=35, color=cmap.colors[0])
                axs[x,y].scatter(orbit[:, 0], orbit[:, 2], c=orbit[:, 2], s = 6, cmap='plasma', alpha=0.5)
                axs[x,y].set_xlabel("X")
                axs[x,y].set_ylabel("Z")
            else:
                axs[x,y].plot(orbit[0, 1], orbit[0, 2], '+', markersize=35, color=cmap.colors[0])
                axs[x,y].scatter(orbit[:, 1], orbit[:, 2], c=orbit[:, 2], s = 6, cmap='plasma', alpha=0.5)
                axs[x,y].set_xlabel("Y")
                axs[x,y].set_ylabel("Z")
        
            axs[x,y].tick_params(labelsize=42)
            axs[x,y].xaxis.label.set_size(42)
            axs[x,y].yaxis.label.set_size(42)
    tight_layout()
    fig.savefig(path, format='png', dpi=400, bbox_inches ='tight', pad_inches = 0.1)
    return

def plot_vf_err(model, dyn_info, model_type, loss_type):
    dyn, dim, time_step = dyn_info
    dyn_sys_type = "lorenz" if dyn == lorenz else "rossler"

    orbit = torchdiffeq.odeint(dyn, torch.randn(dim), torch.arange(0, 5, time_step), method='rk4', rtol=1e-8)
    orbit = torchdiffeq.odeint(dyn, orbit[-1], torch.arange(0, 20, time_step), method='rk4', rtol=1e-8)
    len_o = orbit.shape[0]

    vf_nn = model(0, orbit.to('cuda')).detach().cpu()
    vf = torch.zeros(len_o, dim)
    for i in range(len_o):
        vf[i] = dyn(0,orbit[i])
    vf_nn, vf = vf_nn.T, vf.T
    ax = figure().add_subplot()
    vf_nn, vf = vf_nn.numpy(), vf.numpy()
    mag = np.linalg.norm(vf, axis=0)
    err = np.linalg.norm(vf_nn - vf, axis=0)
    t = time_step*np.arange(0, len_o)
    percentage_err = err/mag*100

    # For debugging purpose, will remove it later
    print("vf_nn", vf_nn.shape)
    print("vf", vf.shape)
    print("vf_nn-vf", vf_nn - vf)
    print("err", err, err.shape)
    print("mag", mag, mag.shape)
    print(percentage_err)
    
    ax.plot(t, percentage_err, "o", label=r"$\frac{\|\hat x - x\|_2}{\|x\|_2}$", ms=3.0)
    np.savetxt(f'{args.train_dir}{args.loss_type}error_attractor.txt', np.column_stack((t, err/mag*100)), fmt='%.6f')
    ax.set_xlabel("time",fontsize=24)
    ax.xaxis.set_tick_params(labelsize=24)
    ax.yaxis.set_tick_params(labelsize=24)
    ax.set_ylim(0, 50)
    ax.legend(fontsize=24)
    ax.grid(True)
    tight_layout()
    path = f"../plot/Relative_error/{args.model_type}_{args.loss_type}_{dyn_sys_type}.png"
    savefig(path)
    return percentage_err

def plot_vf_err_test(model, y_pred_train, dyn_info, model_type, loss_type):
    dyn, dim, time_step = dyn_info
    dyn_sys_type = "lorenz" if dyn == lorenz else "rossler"
    orbit = y_pred_train
    len_o = orbit.shape[0]
    orbit_gpu = orbit.to('cuda')
    vf_nn = model(0, orbit_gpu).detach().cpu()
    vf = torch.zeros(len_o, dim)
    # for i in range(len_o):
    true_vf = lambda x: dyn(0,x)
    vf = torch.vmap(true_vf)(orbit_gpu).detach().cpu()
    vf_nn, vf = vf_nn.T, vf.T
    ax = figure().add_subplot()
    vf_nn, vf = vf_nn.numpy(), vf.numpy()
    mag = np.linalg.norm(vf, axis=0)
    # mag = abs(vf[2])
    err = np.linalg.norm(vf_nn - vf, axis=0)
    # err = abs(vf_nn[2]-vf[2])
    t = time_step*np.arange(0, len_o)
    ax.plot(t, err/mag*100, "o", label=r"$\|Error\|_2$", ms=3.0)
    np.savetxt(f'{args.train_dir}{args.loss_type}error_test.txt', np.column_stack((t, err/mag*100)), fmt='%.6f')
    ax.set_xlabel("time",fontsize=24)
    ax.xaxis.set_tick_params(labelsize=24)
    ax.yaxis.set_tick_params(labelsize=24)
    ax.set_ylim(0, 2)
    ax.legend(fontsize=24)
    ax.grid(True)
    tight_layout()

    path = f"{args.train_dir}MSE_error_Ytest.png"
    savefig(path)

def plot_vector_field(model, path, idx, t, N, device='cuda'):
    # Credit: https://torchdyn.readthedocs.io/en/latest/_modules/torchdyn/utils.html

    x = torch.linspace(-50, 50, N)
    y = torch.linspace(-50, 50, N)
    X, Y = torch.meshgrid(x,y)
    Z_random = torch.randn(1)*10
    U, V = np.zeros((N,N)), np.zeros((N,N))

    for i in range(N):
        for j in range(N):
            if idx == 1:
                phi = torch.stack([X[i,j], Y[i,j], torch.tensor(20.)]).to('cuda')
            else:
                phi = torch.stack([X[i,j], torch.tensor(0), Y[i,j]]).to('cuda')
            O = model(0., phi).detach().cpu().numpy()
            if O.ndim == 1:
                U[i,j], V[i,j] = O[0], O[idx]
            else:
                U[i,j], V[i,j] = O[0, 0], O[0, idx]

    fig = figure(figsize=(5,4))
    ax = fig.add_subplot(111)
    contourf = ax.contourf(X, Y, np.sqrt(U**2 + V**2), cmap='jet')
    ax.streamplot(X.T.numpy(),Y.T.numpy(),U.T,V.T, color='k')
    ax.set_xlim([x.min(),x.max()])
    ax.set_ylim([y.min(),y.max()])
    ax.set_xlabel(r"$x$", fontsize=17)
    if idx == 1:
        ax.set_ylabel(r"$y$", fontsize=17)
    else:
        ax.set_ylabel(r"$z$", fontsize=17)
    ax.xaxis.set_tick_params(labelsize=17)
    ax.yaxis.set_tick_params(labelsize=17)
    fig.colorbar(contourf)
    tight_layout()
    savefig(path, format='jpg', dpi=400, bbox_inches ='tight', pad_inches = 0.1)
    close()
    return

def rk4(x, f, dt):
    k1 = f(0, x)
    k2 = f(0, x + dt*k1/2)
    k3 = f(0, x + dt*k2/2)
    k4 = f(0, x + dt*k3)
    return x + dt/6*(k1 + 2*k2 + 2*k3 + k4)
    
def lyap_exps(dyn_sys_info, traj, iters):
    model, dim, time_step = dyn_sys_info
    LE = torch.zeros(dim).to(device)
    traj_gpu = traj.to(device)
    if model == lorenz:
        f = lambda x: rk4(x, model, time_step)
        Jac = torch.vmap(torch.func.jacrev(f))(traj_gpu)
    else:
        f = model
        traj_in_batch = traj_gpu.reshape(-1, 1, dim, 1)
        print("shape", traj_in_batch.shape)
        Jac = torch.randn(traj_gpu.shape[0], dim, dim).cuda()
        for j in range(traj_in_batch.shape[0]):
            Jac[j] = torch.autograd.functional.jacobian(f, traj_in_batch[j]).squeeze()
        # Not possible due to inplace arithmatic in line 82
        # Jac = torch.vmap(torch.func.jacrev(f))(traj_in_batch)

    Q = torch.rand(dim,dim).to(device)
    eye_cuda = torch.eye(dim).to(device)
    for i in range(iters):
        if i > 0 and i % 1000 == 0:
            print("Iteration: ", i, ", LE[0]: ", LE[0].detach().cpu().numpy()/i/time_step)
        Q = torch.matmul(Jac[i], Q)
        Q, R = torch.linalg.qr(Q)
        LE += torch.log(abs(torch.diag(R)))
    return LE/iters/time_step


if __name__ == '__main__':

    # Set device
    torch.manual_seed(42)
    torch.cuda.empty_cache()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("device: ", device)

    parser = argparse.ArgumentParser()
    parser.add_argument("--time_step", type=float, default=1e-2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--num_epoch", type=int, default=1000)
    parser.add_argument("--num_train", type=int, default=5000)
    parser.add_argument("--num_test", type=int, default=2000)
    parser.add_argument("--num_val", type=int, default=0)
    parser.add_argument("--num_trans", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--loss_type", default="Sobolev", choices=["Jacobian", "MSE", "Sobolev"])
    parser.add_argument("--dyn_sys", default="lorenz", choices=["lorenz", "rossler"])
    parser.add_argument("--model_type", default="MLP_skip", choices=["MLP","MLP_skip", "CNN", "HigherDimCNN", "GRU"])
    parser.add_argument("--optim_name", default="AdamW", choices=["AdamW", "Adam", "RMSprop", "SGD"])
    parser.add_argument("--n_hidden", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--reg_param", type=float, default=500)
    parser.add_argument("--train_dir", default="../plot/Vector_field/")

    # Initialize Settings
    args = parser.parse_args()
    if not os.path.exists(args.train_dir):
        os.makedirs(args.train_dir)
    dim = 3
    dyn_sys_func = lorenz if args.dyn_sys == "lorenz" else rossler
    dyn_sys_info = [dyn_sys_func, dim, args.time_step]
    criterion = torch.nn.MSELoss()

    # Save initial settings
    start_time = datetime.datetime.now().strftime("%m_%d_%H_%M_%S")
    out_file = os.path.join("../test_result/", f"{start_time}_{args.model_type}_{args.loss_type}_{args.dyn_sys}.txt")
    logging.basicConfig(filename=out_file, level=logging.INFO, format="%(message)s")
    logger = logging.getLogger()
    for arg, value in vars(args).items():
        logger.info("%s: %s", arg, value)

    # Create Dataset
    dataset = create_data(dyn_sys_info, n_train=args.num_train, n_test=args.num_test, n_trans=args.num_trans, n_val=args.num_val)

    # Create model
    m = FNO1d(modes=2, width=args.n_hidden).cuda()
    print("num_param", count_params(m))

    print("Training...") # Train the model, return node
    epochs, loss_hist, test_loss_hist, jac_train_hist, jac_test_hist, Y_test = train(dyn_sys_info, m, device, dataset, args.optim_name, criterion, args.num_epoch, args.lr, args.weight_decay, args.reg_param, args.loss_type, args.model_type, args.batch_size)

    # Plot Loss
    loss_path = f"../plot/Loss/{args.dyn_sys}/{args.model_type}_{args.loss_type}_Total_{start_time}.png"
    jac_loss_path = f"../plot/Loss/{args.dyn_sys}/{args.model_type}_{args.loss_type}_Jacobian_matching_{start_time}.png"
    mse_loss_path = f"../plot/Loss/{args.dyn_sys}/{args.model_type}_{args.loss_type}_MSE_part_{start_time}.png"
    true_plot_path_1 = f"../plot/Vector_field/True_{args.dyn_sys}_1.png"
    true_plot_path_2 = f"../plot/Vector_field/True_{args.dyn_sys}_2.png"
    phase_path = f"../plot/Phase_plot/{args.dyn_sys}_{args.model_type}_{args.loss_type}.png"

    plot_loss(epochs, loss_hist, test_loss_hist, loss_path) 
    if args.loss_type == "Jacobian":
        plot_loss(epochs, jac_train_hist, jac_test_hist, jac_loss_path) 
        plot_loss(epochs, abs(loss_hist - args.reg_param*jac_train_hist)*(args.time_step)**2, abs(test_loss_hist - args.reg_param*jac_test_hist)*(args.time_step)**2, mse_loss_path) 

    # Plot vector field & phase space
    # percentage_err = plot_vf_err(m, dyn_sys_info, args.model_type, args.loss_type)
    # plot_vf_err_test(m, Y_test, dyn_sys_info, args.model_type, args.loss_type)
    # plot_vector_field(dyn_sys_func, path=true_plot_path_1, idx=1, t=0., N=100, device='cuda')
    # plot_vector_field(dyn_sys_func, path=true_plot_path_2, idx=2, t=0., N=100, device='cuda')
    plot_attractor(m, dyn_sys_info, 50, phase_path)

    # compute LE
    init = torch.randn(dim)
    true_traj = torchdiffeq.odeint(dyn_sys_func, torch.randn(dim), torch.arange(0, 50, args.time_step), method='rk4', rtol=1e-8)
    print("Computing LEs of NN...")
    learned_LE = lyap_exps([m, dim, args.time_step], true_traj, true_traj.shape[0]).detach().cpu().numpy()
    print("Computing true LEs...")
    True_LE = lyap_exps(dyn_sys_info, true_traj, true_traj.shape[0]).detach().cpu().numpy()
    loss_hist, test_loss_hist, jac_train_hist, jac_test_hist

    logger.info("%s: %s", "Training Loss", str(loss_hist[-1]))
    logger.info("%s: %s", "Test Loss", str(test_loss_hist[-1]))
    if args.loss_type == "Jacobian":
        logger.info("%s: %s", "Jacobian term Training Loss", str(jac_train_hist[-1]))
        logger.info("%s: %s", "Jacobian term Test Loss", str(jac_test_hist[-1]))
    logger.info("%s: %s", "Learned LE", str(learned_LE))
    logger.info("%s: %s", "True LE", str(True_LE))
    # logger.info("%s: %s", "Relative Error", str(percentage_err))
    print("Learned:", learned_LE, "\n", "True:", True_LE)
