import numpy as np
import matplotlib

class BlackBox1D:
    def __init__(self):
        pass
    
    @staticmethod
    def FunctionA(x):
        return (1+np.sin(.5*np.sin(x-6)*x + 4))*(-x*x/3 + 4*x - 1)+np.random.normal(0, .05, 1)
    
    @staticmethod
    def Sinc(x):
        sc = 1
        if (x != 4):
            sc = np.sin(x-4)/(x-4)
            
        return  5*sc + np.random.normal(0, .1, 1)



class BlackBox2D:
    def __init__(self):
        pass
    
    @staticmethod
    def FunctionA(x):
        return BlackBox1D.FunctionA(x[0]) + BlackBox1D.Sinc(x[1])
    
    
class BlackBoxND:
    def __init__(self):
        pass
    
    @staticmethod
    def sinND(x):
        return sum(np.sin([i/2 for i in x])) + np.random.normal(0, .005, 1)