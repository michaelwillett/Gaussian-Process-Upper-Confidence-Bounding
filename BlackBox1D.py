import numpy as np
import matplotlib

class FunctionA:
    def __init__(self):
        pass
        
    def evaluate(self, x):
        return (1+np.sin(.5*np.sin(x-6)*x + 4))*(-x*x/3 + 4*x - 1)+np.random.normal(0, .5, 1)
    
    
    
    
        