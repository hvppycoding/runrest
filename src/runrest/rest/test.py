import sys
import numpy as np
import torch
import os
import time
from runrest.rest.models.actor_critic import Actor
from runrest.rest.utils.rsmt_utils import *
from runrest.rest.utils.log_utils import *
import math
import argparse

if sys.version_info < (3, 9):
    # importlib.resources either doesn't exist or lacks the files()
    # function, so use the PyPI version:
    import importlib_resources
else:
    # importlib.resources has files(), so use that:
    import importlib.resources as importlib_resources

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--degree', type=int, default=10, help='maximum degree of nets')
parser.add_argument('--dimension', type=int, default=2, help='terminal representation dimension')
parser.add_argument('--test_data', type=str, default='', help='test data')
parser.add_argument('--test_size', type=int, default=10000, help='number of nets')
parser.add_argument('--batch_size', type=int, default=1000, help='test batch size')
parser.add_argument('--transformation', type=int, default=1, help='number of transformations for inference')
# parser.add_argument('--run_optimal', type=str, default='true', help='run GeoSteiner to generate optimal RSMT')
parser.add_argument('--run_optimal', type=str, default='false', help='run GeoSteiner to generate optimal RSMT')
parser.add_argument('--plot_first', type=str, default='true', help='plot the first result')
parser.add_argument('--seed', type=int, default=7, help='random seed')

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device("cpu")

print()

ckp_dir = "runrest.rest.checkpoints"
ckp_file = "rsmt" + str(args.degree) + "b.pt"

ckp_resource = importlib_resources.files(ckp_dir).joinpath(ckp_file)
checkpoint = torch.load(ckp_resource, map_location=device)

actor = Actor(args.degree, device)
actor.load_state_dict(checkpoint['actor_state_dict'], )
actor.eval()
evaluator = Evaluator(degree=args.degree)

if os.path.exists(args.test_data):
    test_cases = read_data(args.test_data)
else:
    np.random.seed(args.seed)
    test_cases = np.random.rand(args.test_size, args.degree, args.dimension)
    test_cases = np.round(test_cases, 8)

num_batches = (args.test_size + args.batch_size - 1) // args.batch_size

start_time = time.time()
if args.transformation <= 1:
    all_outputs = []
    for b in range(num_batches):
        test_batch = test_cases[b * args.batch_size : (b+1) * args.batch_size]
        with torch.no_grad():
            outputs, _ = actor(test_batch, True)
        all_outputs.append(outputs.cpu().detach().numpy())
    inference_time = time.time() - start_time

    all_outputs = np.concatenate(all_outputs, 0)
    mean_length = 0
    all_lengths = evaluator.eval_batch(test_cases, all_outputs, args.degree)
else:
    inference_time = 0
    all_lengths = []
    all_outputs = []
    for b in range(num_batches):
        test_batch = test_cases[b * args.batch_size : (b+1) * args.batch_size]
        best_lengths = [1e9 for i in range(len(test_batch))]
        best_outputs = [[] for i in range(len(test_batch))]
        for t in range(args.transformation):
            transformed_batch = transform_inputs(test_batch, t)
            ttime = time.time()
            with torch.no_grad():
                outputs, _ = actor(transformed_batch, True)
            inference_time += time.time() - ttime
            outputs = outputs.cpu().detach().numpy()
            lengths = evaluator.eval_batch(transformed_batch, outputs, args.degree)
            if t >= 4:
                outputs = np.flip(outputs, 1)
            for i in range(len(test_batch)):
                if lengths[i] < best_lengths[i]:
                    best_lengths[i] = lengths[i]
                    best_outputs[i] = outputs[i]
                
        all_lengths.append(best_lengths)
        all_outputs.append(best_outputs)
    all_lengths = np.concatenate(all_lengths, 0)
    all_outputs = np.concatenate(all_outputs, 0)
    
full_time = time.time() - start_time

print('REST mean length       ', round(all_lengths.mean(), 6))
print('inference time         ', round(inference_time, 3))
print('   full   time         ', round(full_time, 3))
print()

if args.plot_first.lower() == 'true':
    # Index 0
    fig = plt.figure(figsize=(10, 4.6))
    # REST solution
    plot_rest(test_cases[0], all_outputs[0])
    plt.annotate('REST ' + str(round(all_lengths[0], 3)), (-0.04, -0.04))
    plt.show()
    
    # Index 1
    fig = plt.figure(figsize=(10, 4.6))
    # REST solution
    plot_rest(test_cases[1], all_outputs[1])
    plt.annotate('REST ' + str(round(all_lengths[1], 3)), (-0.04, -0.04))
    plt.show()