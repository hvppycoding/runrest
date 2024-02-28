import os
from typing import List
import numpy as np


class Parser:
    def readlines(self, file) -> None:
        if not os.path.exists(file):
            raise FileNotFoundError('File not found: ' + file)
        with open(file, 'r') as f:
            lines = f.readlines()
            return lines
            
    def parselines(self, lines: List[str]) -> List[List[float]]:
        pointslist: List[List[float]] = []
        for line in lines:
            line = line.strip().split(' ')
            if len(line) % 2 != 0:
                raise ValueError('Invalid number of coordinates: ' + str(len(line)))
            points = np.array([float(coord) for coord in line])
            points = points.reshape([-1, 2])
            pointslist.append(points.tolist())
        return pointslist
    
    def parse(self, file: str) -> List[List[float]]:
        lines = self.readlines(file)
        pointslist = self.parselines(lines)
        return pointslist
        

def parse_input_file(file: str) -> List[List[float]]:
    return Parser().parse(file)