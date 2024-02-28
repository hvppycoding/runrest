# RSMT utils
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import ctypes
import heapq


class Evaluator:
    def __init__(self, degree):
        self.degree = degree
        self.data = np.zeros((degree, 6), dtype=np.float32)
        
    def eval_func(self, inputs, outputs, degree):
        # 0: x
        # 1: y
        # 2: xl
        # 3: yl
        # 4: xr
        # 5: yh

        self.data[:, 0::2] = inputs[:, 0, None]
        self.data[:, 1::2] = inputs[:, 1, None]
        
        for i in range(int(len(outputs) / 2)):
            n1 = outputs[2*i]
            n2 = outputs[2*i+1]
            x = self.data[n1, 0]
            y = self.data[n2, 1]
            self.data[n2, 2] = min(self.data[n2, 2], x)
            self.data[n2, 4] = max(self.data[n2, 4], x)
            self.data[n1, 3] = min(self.data[n1, 3], y)
            self.data[n1, 5] = max(self.data[n1, 5], y)
        return np.sum(self.data[:, 4] - self.data[:, 2] + self.data[:, 5] - self.data[:, 3])
        
    def eval_batch(self, input_batch, output_batch, degree):
        lengths = []
        batch_size = len(input_batch)
        for i in range(batch_size):
            lengths.append(self.eval_func(input_batch[i], output_batch[i], degree))
        return np.array(lengths)

edge_color = 'black'
edge_width = .5
term_color = 'black'
term_size = 4
steiner_color = 'black'

def plot_rest(input, output):
    input = np.array(input)
    output = np.array(output)
    x_low, y_low, x_high, y_high = [], [], [], []
    for i in range(len(input)):
        x_low.append(input[i][0])
        y_low.append(input[i][1])
        x_high.append(input[i][0])
        y_high.append(input[i][1])
    for i in range(int(len(output) / 2)):
        x_idx = output[2*i]
        y_idx = output[2*i+1]
        x = input[x_idx][0]
        y = input[y_idx][1]
        y_low[x_idx] = min(y_low[x_idx], y)
        y_high[x_idx] = max(y_high[x_idx], y)
        x_low[y_idx] = min(x_low[y_idx], x)
        x_high[y_idx] = max(x_high[y_idx], x)
    for i in range(len(x_low)):
        plt.plot([x_low[i], x_high[i]], [input[i][1], input[i][1]], '-', color=edge_color, linewidth=edge_width)
        plt.plot([input[i][0], input[i][0]], [y_low[i], y_high[i]], '-', color=edge_color, linewidth=edge_width)
    plt.plot(list(input[:,0]), list(input[:,1]), 's', color=term_color, markerfacecolor='black', markersize=term_size, markeredgewidth=edge_width)
    for idx, (x, y) in enumerate(input.tolist()):
        plt.text(x, y, str(idx), fontsize=12, ha='right')
    ax = plt.gca()
    ax.set_aspect('equal')
    xmax = max(x_high)
    ymax = max(y_high)
    xmin = min(x_low)
    ymin = min(y_low)
    plt.xlim(xmin - 0.05 * (xmax - xmin), xmax + 0.05 * (xmax - xmin))
    plt.ylim(ymin - 0.05 * (ymax - ymin), ymax + 0.05 * (ymax - ymin))
    plt.savefig('rsmt.pdf')

def plot_gst_rsmt(terms, sps, edges):
    degree = len(terms)
    points = np.concatenate([terms, sps], 0)
    for i in range(len(edges)):
        u = edges[i][0]
        v = edges[i][1]
        plt.plot([points[u][0], points[u][0]], [points[u][1], points[v][1]], '-', color=edge_color, linewidth=edge_width)
        plt.plot([points[u][0], points[v][0]], [points[v][1], points[v][1]], '-', color=edge_color, linewidth=edge_width)
    plt.plot([terms[i][0] for i in range(degree)], [terms[i][1] for i in range(degree)], 's', markerfacecolor='black', color=term_color, markersize=term_size, markeredgewidth=edge_width)
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)        
    
def save_data(test_data, test_file):
    with open(test_file, 'w') as f:
        for data in test_data:
            f.write(' '.join(['{:.8f} {:.8f}'.format(term[0], term[1]) for term in data]))
            f.write('\n')
            
def read_data(test_file):
    with open(test_file, 'r') as f:
        test_data = []
        lines = f.readlines()
        for line in lines:
            line = line.strip().split(' ')
            data = [float(coord) for coord in line]
            data = np.array(data).reshape([-1, 2])
            test_data.append(data)
    return np.array(test_data)
        

def transform_inputs(inputs, t):
    # 0 <= t <= 7
    xs = inputs[:,:,0]
    ys = inputs[:,:,1]
    if t >= 4:
        temp = xs
        xs = ys
        ys = temp
    if t % 2 == 1:
        xs = 1 - xs
    if t % 4 >= 2:
        ys = 1 - ys
    return np.stack([xs, ys], -1)
