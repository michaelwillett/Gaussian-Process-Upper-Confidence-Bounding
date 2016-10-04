'''
Created on Oct 1, 2016

@author: Michael Willett
'''
import pyGPs
import numpy as np
import BlackBox1D
import time

if __name__ == '__main__':
    sample = BlackBox1D.FunctionA()
    
    br = False
    class domain:
        def __init__(self):
            pass
     
    d = domain()
        
    d.min = [0]
    d.max = [10]
    
    x = np.array([sum(i)/2. for i in zip(d.min,d.max)])
    y = np.array(sample.evaluate(x))
    z = np.linspace(d.min[0],d.max[0])
    
    
    model = pyGPs.GPR()      # specify model (GP regression)
    k = pyGPs.cov.Matern()
    m = pyGPs.mean.Linear()
    
    converged = False
    t = 1
    delta = .5
    
    #for i in range(0, 10):
    while not converged:
        model.getPosterior(x, y) # fit default model (mean zero & rbf kernel) with data
        model.optimize(x, y)     # optimize hyperparamters (default optimizer: single run minimize)
        model.predict(z)         # predict test cases
        
        mu_t = [item for sublist in model.ym for item in sublist] # compress from list of 1-element lists to list
        s2_t = [item for sublist in model.ys2 for item in sublist] # compress from list of 1-element lists to list
        
        beta_t = 2*np.log(t*t*np.pi*np.pi/(6*delta))
        
        rng = [a+b*beta_t for a,b in zip(mu_t, s2_t)]
        x_t = z[np.argmax(rng)]
        
        converged = (x_t == x[-1]) or (t > 100)
        t += 1
        print x_t, "<", t, ">"
                
        
        x = np.append(x, x_t)
        y = np.append(y, sample.evaluate(x_t))
        
        
    print "estimated max: ", x[-1]
    model.plot()             # and plot result