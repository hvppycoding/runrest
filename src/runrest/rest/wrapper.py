from numbers import Number
import numpy as np
import sys
import numpy as np
import numpy.typing as npt
import torch
import os
import time
from typing import List
from runrest.rest.models.actor_critic import Actor
from runrest.rest.utils.rsmt_utils import *
from runrest.rest.utils.log_utils import *


if sys.version_info < (3, 9):
    # importlib.resources either doesn't exist or lacks the files()
    # function, so use the PyPI version:
    import importlib_resources
else:
    # importlib.resources has files(), so use that:
    import importlib.resources as importlib_resources

def scale_data(arr: npt.NDArray) -> npt.NDArray:
    arr = np.array(arr, dtype=np.float32)
    max_values = np.max(arr, axis=1)
    min_values = np.min(arr, axis=1)
    mid_values = (max_values + min_values) / 2
    range_values = np.max(max_values - min_values, axis=1) / 0.9
    return (arr - mid_values[:, None, :]) / range_values[:, None, None] + 0.5

def run_rest_2pin(input_data: List[List[Number]]) -> List[List[Number]]:
    return [[0, 1] for _ in input_data]

def run_rest_same_degree(input_data: List[List[List[Number]]], degree: int, transformation: int = 1) -> List[List[Number]]:
    BATCH_SIZE = 128
    available_degrees = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    closest_degree = min(available_degrees, key=lambda x:abs(x-degree))
    ckp_dir = "runrest.rest.checkpoints"
    ckp_file = "rsmt" + str(closest_degree) + "b.pt"
    ckp_resource = importlib_resources.files(ckp_dir).joinpath(ckp_file)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(ckp_resource, map_location=device)
    actor = Actor(degree, device)
    actor.load_state_dict(checkpoint['actor_state_dict'])
    actor.eval()
    evaluator = Evaluator(degree=degree)

    original_test_cases = np.array(input_data, dtype=np.float32)
    test_cases = scale_data(original_test_cases)
    
    num_batches = (len(input_data) + BATCH_SIZE - 1) // BATCH_SIZE

    start_time = time.time()
    if transformation <= 1:
        all_outputs = []
        for b in range(num_batches):
            test_batch = test_cases[b * BATCH_SIZE : (b+1) * BATCH_SIZE]
            with torch.no_grad():
                outputs, _ = actor(test_batch, True)
            all_outputs.append(outputs.cpu().detach().numpy())
        inference_time = time.time() - start_time

        all_outputs = np.concatenate(all_outputs, 0)
        mean_length = 0
        all_lengths = evaluator.eval_batch(test_cases, all_outputs, degree)
    else:
        inference_time = 0
        all_lengths = []
        all_outputs = []
        for b in range(num_batches):
            test_batch = test_cases[b * BATCH_SIZE : (b+1) * BATCH_SIZE]
            best_lengths = [1e9 for i in range(len(test_batch))]
            best_outputs = [[] for i in range(len(test_batch))]
            for t in range(transformation):
                transformed_batch = transform_inputs(test_batch, t)
                ttime = time.time()
                with torch.no_grad():
                    outputs, _ = actor(transformed_batch, True)
                inference_time += time.time() - ttime
                outputs = outputs.cpu().detach().numpy()
                lengths = evaluator.eval_batch(transformed_batch, outputs, degree)
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
    return all_outputs.tolist()
    
def run_rest(input_data: List[List[List[Number]]], heuristic_2pin: bool = False) -> List[List[Number]]:
    degree_to_index: dict[int, List[int]] = {}
    for i in range(len(input_data)):
        degree = len(input_data[i])
        if degree not in degree_to_index:
            degree_to_index[degree] = []
        degree_to_index[degree].append(i)
    
    output_data: List[List[Number]] = [[] for i in range(len(input_data))]
    
    for degree in degree_to_index:
        input_same_degree: List[List[List[Number]]] = []
        for idx in degree_to_index[degree]:
            input_same_degree.append(input_data[idx])
            
        if degree == 2 and heuristic_2pin:
            outputs = run_rest_2pin(input_same_degree)
        else:
            outputs = run_rest_same_degree(input_same_degree, degree)
        
        for idx, output in zip(degree_to_index[degree], outputs):
            output_data[idx] = output
    
    return output_data

def test_scaler():
    test_cases = np.random.rand(10, 5, 2)
    
    for i in range(10):
        test_cases[i, :, 0] *= i + 1
        test_cases[i, :, 1] *= 10 - i

    scaled_cases = scale_data(test_cases)
    
    import matplotlib.pyplot as plt
    for i in range(10):
        plt.subplot(1, 2, 1)
        plt.scatter(test_cases[i, :, 0], test_cases[i, :, 1], color='black', marker='s')
        plt.xlim(0, 10)
        plt.ylim(0, 10)
        ax = plt.gca()
        ax.set_aspect('equal')
        
        plt.subplot(1, 2, 2)
        plt.scatter(scaled_cases[i, :, 0], scaled_cases[i, :, 1], color='black', marker='s')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        ax = plt.gca()
        ax.set_aspect('equal')
        
        plt.show()
        
def test_run_rest():
    test_cases = np.random.rand(10, 10, 2)
    
    for i in range(10):
        test_cases[i, :, 0] *= i + 1
        test_cases[i, :, 1] *= 10 - i
        
    test_cases = test_cases.tolist()
        
    for i in range(10):
        test_cases[i] = test_cases[i][min(i, 8):]
    outputs = run_rest(test_cases)
    
    for i in range(10):
        plot_rest(np.array(test_cases[i]), np.array(outputs[i]))
        plt.show()
        
def test_run_rest_2pin():
    test_cases = np.random.rand(10, 2, 2)
    test_cases = test_cases.tolist()
    outputs = run_rest(test_cases, heuristic_2pin=True)
    for i in range(10):
        plot_rest(np.array(test_cases[i]), np.array(outputs[i]))
        plt.show()

if __name__ == "__main__":
    # test_scaler()
    test_run_rest_2pin()
    test_run_rest()
