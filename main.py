import torch
import wandb
from transform import Flow
from toydata import ToyData
from torch.utils.data import DataLoader, random_split
from utils import make_toy_graph
from torchviz import make_dot
import pyro
import matplotlib.pyplot as plt
import tqdm
from distributions import DoubleDistribution, StandardNormal, Normal, SemanticDistribution
import affine_coupling
from permuters import LinearLU, Shuffle, Reverse
import torch.nn as nn
from affine_coupling import AffineCoupling
import torch.optim as optim
import torch.distributions as dist
from act_norm import ActNormBijectionCloud, ActNormBijection

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


run = wandb.init(project='toy_data_4.0', entity='mvalente',
                 config=r'config/base_conf.yaml')

smoke_test = False
init_config = wandb.config

points_per_sample = 30000
run.config.points_per_sample = points_per_sample

device = "cpu"
toy_data = ToyData(points_per_sample, show_data=False)
generate_all = True


cs = toy_data.cs
cu = toy_data.cu
contexts = torch.vstack((cs, cu))

input_dim = 4
context_dim = 2
split_dim = input_dim - context_dim

train_size = int(toy_data.__len__() * 0.9)
test_size = toy_data.__len__() - train_size
train_set, test_set = random_split(toy_data,
                                   [train_size, test_size],
                                   generator=torch.Generator(device="cpu"))

train_loader = DataLoader(train_set,
                          batch_size=init_config.batch_size,
                          shuffle=False)

test_seen = train_loader.dataset.dataset.x.clone()
test_seen = [t for t in torch.split(test_seen,
                                    points_per_sample)]


visual_distribution = dist.MultivariateNormal(torch.zeros(split_dim), torch.eye(split_dim))
semantic_distribution = SemanticDistribution(contexts.float(), torch.ones(context_dim), (2, 1))

base_dist = DoubleDistribution(visual_distribution, semantic_distribution, input_dim, context_dim)

permuter = lambda dim: LinearLU(num_features=dim, eps=1.0e-5)
# permuter = lambda dim: Reverse(dim_size=dim)

transforms = []
# transforms.append(ActNormBijectionCloud(input_dim, data_dep_init=True))
for index in range(init_config.block_size):
    transforms.append(AffineCoupling(input_dim, hidden_dims=[2], non_linearity=nn.LeakyReLU()))
    if index != init_config.block_size - 1:
        transforms.append(ActNormBijection(input_dim, data_dep_init=True))
        transforms.append(permuter(input_dim))

flow = Flow(transforms, base_dist)
flow.train()
flow = flow.to(device)

print(f'Number of trainable parameters: {sum([x.numel() for x in flow.parameters()])}')
run.watch(flow)
optimizer = optim.Adam(flow.parameters(), lr=init_config.lr)  # todo tune WD

epochs = tqdm.trange(1, init_config.epochs) if not smoke_test else [1, 2]

number_samples = 400
for epoch in epochs:
    losses = []
    losses_flow = []
    losses_centr = []
    losses_mmd = []
    for data, targets in train_loader:
        data = data.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        loss_flow = - flow.log_prob(data, targets).mean() * 2
        centralizing_loss = flow.centralizing_loss(data, targets, cs.to(device))
        mmd_loss = flow.mmd_loss(data, cu.to(device)) * 0.1
        loss = loss_flow + centralizing_loss + mmd_loss
        loss.backward()
        optimizer.step()

        losses_flow.append(loss_flow.item())
        losses_centr.append(centralizing_loss.item())
        losses_mmd.append(mmd_loss.item())
        losses.append(loss.item())

    if epoch % 2 == 0:
        with torch.no_grad():
            test_data = []
            if generate_all:
                for c_id, c in enumerate(contexts):
                    test_data.append(flow.generation(
                                     torch.hstack((c.repeat(number_samples).reshape(-1, 2),
                                                  flow.base_dist.visual_distribution.sample([number_samples])))))
            else:
                number_samples = points_per_sample
                test_data = test_seen.copy()
                test_data.append(flow.generation(
                                 torch.hstack((cu.repeat(number_samples).reshape(-1, 2),
                                               flow.base_dist.visual_distribution.sample([number_samples])))))

                test_data = [data.to("cpu").detach().numpy() for data in test_data]

            make_toy_graph(test_data, epoch, fit=False, show=False, save=True, path='plots/t2/')
    
    run.log({"loss": sum(losses) / len(losses),
             "loss_flow": sum(losses_flow) / len(losses_flow),  # }, step=epoch)
             "loss_central": sum(losses_centr) / len(losses_centr),  # }, step=epoch)
             "loss_mmd": sum(losses_mmd) / len(losses_mmd)}, step=epoch)

    if loss.isnan():
        print('Nan in loss!')
        Exception('Nan in loss!')
       
