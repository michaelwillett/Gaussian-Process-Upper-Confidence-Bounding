'''
Created on Oct 1, 2016

@author: Michael Willett
'''
import pyGPs
import itertools as it
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
import BlackBox
import time

class GP_UCB:
    def __init__(self):
        self.func = None
        self.model = None
        self.domain = []
        self.dim = 0
        self.max_t = 0
        
    def minfunc(self,x):
        self.model.predict(np.array([x]))
        t = self.regret.size
        t = 1
        
        # Get information gain
        if model.kernel == 'Linear':
            self.IG = model.dim * np.sqrt(t)
            
        if model.kernel == 'RBF':
            self.IG = np.sqrt(t*pow(np.log(t),self.dim+1))
            
        if model.kernel == 'Matern':
            v = 1.5
            e = (v + model.dim*(model.dim + 1)) / (2*v + model.dim*(model.dim + 1))
            self.IG = pow(t, e)

        delta = .9      # delta \in (0,1): the lower this is, the more we explore uncertainty
        normD = np.linalg.norm([x[1]-x[0] for x in self.domain])
        self.beta_t = 2*np.log(self.dim*t*t*np.pi*np.pi/(6*delta)) 
        
        #self.beta_t = self.IG*pow(np.log(t/delta),3)
        
        mu_t = [item for sublist in self.model.ym for item in sublist] # compress from list of 1-element lists to list
        s2_t = [item for sublist in self.model.ys2 for item in sublist] # compress from list of 1-element lists to list
                    
        retv = -1*(mu_t[0]+s2_t[0]*np.sqrt(self.beta_t))
        return  retv
    
    def UCB(self): 
        x0 = [[sum(n)/2 for n in self.domain]]
        
        params = {'approx_grad':True} 
        kwargs = {"method": "TNC", "bounds": self.domain, "options":params}
        rslt = opt.basinhopping(self.minfunc, x0, minimizer_kwargs=kwargs, niter=3)
        
        return rslt.x
    
    def TestConvergence(self, x):
        #return  (np.linalg.norm(x[-1]-x[-2]) < .001)
        return self.regret.size > model.max_t
    
    
    def Plot2D(self, x, reset = False):
        self.model.predict(self.z)
        
        mp = np.reshape(self.model.ym, (self.res[0], self.res[1]))
        up = np.reshape(self.model.ym + self.model.ys2, (self.res[0], self.res[1]))
        dp = np.reshape(self.model.ym - self.model.ys2, (self.res[0], self.res[1]))
        xp = np.reshape(self.z[:,0], (self.res[0], self.res[1]))
        yp = np.reshape(self.z[:,1], (self.res[0], self.res[1]))
                
        self.surf = self.ax.plot_surface(xp, yp, mp, color='blue', zorder=0, alpha=0.5)
        self.up = self.ax.plot_wireframe(xp, yp, up, color='green', rstride=5, cstride=5, zorder=1)
        self.dp = self.ax.plot_wireframe(xp, yp, dp, color='green', rstride=5, cstride=5, zorder=2)
        
        z = self.model.predict(np.reshape(x, [1,2]))
        self.pt = self.ax.scatter(x[0], x[1], z[0], marker='v', c='red', s=80, zorder=3)
        plt.pause(0.001)
        
        if reset:
            self.surf.remove()
            self.up.remove()
            self.dp.remove()
            self.pt.remove()
        else:
            plt.ioff()
            plt.show()
            
    def PlotRegret(self, reset = False):
        
        T = self.regret.size+1

        s2 = [item for sublist in self.model.ys2 for item in sublist]
        isg = 1/s2[0] 
        regretBound = np.sqrt(T*self.beta_t*self.IG*8/np.log(1+isg))
        
        self.regretBound = np.append(self.regretBound, regretBound)
        
        self.regplt = plt.plot(self.regret, color='b')
        #self.boundplt = plt.plot(self.regretBound, color='r')
        plt.pause(0.001)
        
        if reset:
            self.regplt.pop(0).remove()
            #self.boundplt.pop(0).remove()
        else:
            plt.ioff()
            plt.show()
            
    def PrintStatus(self, modelTime, gradTime):
        l = self.regret.size
        totalTime = round(modelTime + gradTime, 4)*1000
        ratio = round(gradTime*1000 / totalTime, 3)*100
        formatPoint = [ round(x, 4) for x in list(self.x[l-1]) ]
        
        print "Iteration " + `l` + ":"
        print "    sample point:  " + `formatPoint`
        print "    regret:        " + `round(self.regret[l-1]-self.regret[l-2],4)`
        print "    runtime:       " + `totalTime` + \
              "ms (decent: " + `ratio` + "%)"
    
    def Init1D(self):
        self.func = BlackBox.BlackBox1D.FunctionA
        self.domain = [(0, 10)]
        self.dim = 1
        self.max_t = 1000
        self.res = [200]
        self.z = np.array(np.linspace(self.domain[0][0], self.domain[0][1], self.res[0]))
    
    def Init2D(self):
        self.func = BlackBox.BlackBoxND.sinND
        self.domain = [(0, 10), (0, 10)]
        self.dim = 2
        self.max_t = 1000
        self.res = [100, 100]
        x = [np.linspace(self.domain[i][0], self.domain[i][1], self.res[i]) for i in range(self.dim)]
        self.z = np.array(list(it.product(x[0], x[1])))
                
        plt.ion()
        fig = plt.figure()
        self.ax = fig.add_subplot(111, projection='2d')
        
    def InitND(self, kernel='Linear'):
        self.func = BlackBox.BlackBoxND.sinND
        self.domain = [(0, 10), (0, 10), (0, 10), (0, 10), (0, 10), (0, 10), (0, 10), (0, 10)]
        self.optima = [3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416]
        self.dim = 5
        self.max_t = 5000
        self.res = [11, 11, 11, 11, 11, 11, 11, 11]
        x = [np.linspace(self.domain[i][0], self.domain[i][1], self.res[i]) for i in range(self.dim)]
        self.z = np.array(list(it.product(x[0], x[1], x[2])))
        self.kernel = kernel
        
        plt.ion()
        plt.figure()
    
    def GetOptima(self):
        converged = False
        
        # initialize search space
        self.x = np.array([[sum(n)/2 for n in self.domain]])
        self.y = np.array([self.func(self.x[0])])
        self.regret = np.array([np.linalg.norm(self.x[0] - self.optima)])
        self.regretBound = np.array([1])
        
        # specify model (GP regression)
        self.model = pyGPs.GPR()
        m = pyGPs.mean.Zero()
        k = pyGPs.cov.Linear()
        if model.kernel == 'RBF':
            k = pyGPs.cov.RBF()
            
        if model.kernel == 'Matern':
            k = pyGPs.cov.Matern()
        
        self.model.setPrior(mean=m, kernel=k)
        

        #self.model.setNoise(0)
        
        while not converged:
            start = time.time()
            rslt = self.model.getPosterior(self.x, self.y, False) # fit default model (mean zero & rbf kernel) with data
            self.model.optimize(self.x, self.y)     # optimize hyperparamters (default optimizer: single run minimize)
            end = time.time()
            
            self.post = rslt[1]
            
            x_t = self.UCB()
            self.regret = np.append(self.regret, np.linalg.norm(x_t - self.optima) 
                                    + self.regret[self.regret.size-1])

            
            self.x = np.append(self.x, np.array([x_t]), axis=0)
            self.y = np.append(self.y, self.func(x_t))
            
            converged = self.TestConvergence(self.x)
            self.PrintStatus(end-start, time.time()-end)
            self.PlotRegret(True)

        
        self.PlotRegret(False)
        #self.Plot2D(x_t)
        
if __name__ == '__main__':
    model = GP_UCB()
    model.InitND('RBF')
    model.GetOptima()