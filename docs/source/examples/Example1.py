#!/usr/bin/env python
# coding: utf-8

# # Example 1
# 
# This is an example that reproduces Figure 3 in the main text (up to iteration 50).
# It also calculates validated negative loglikelihood at each iteration. 

# In[1]:


import neuralflow
import numpy as np
import matplotlib.pyplot as plt, matplotlib.gridspec as gridspec


# ## Step 1: Generate synthetic data
# 
# 1) Specify the ground-truth model for data generation, see the implementation of the `EnergyModel` class for available options.
# Here we use the spectral elements method (SEM) for the eigenvector-eigenvalue problem with `Ne=256` elements and `Np=8` points per element. We retain `Nv=64` eigenvectors and eigenvalues of the $\mathcal{H}$ operator for the likelihood and gradients calculations. We use double-well model (as in main text FIG. 3, 4), noise magnitude `D0=10`, 1 neural response with the firing rate function `f(x) = 100*(x+1)` hz. We represent the results by plotting model potential, use equilibrium probability distribution peq$=\exp(-\Phi(x))$ to save the results, and optimize the driving force $F(x)$. All these quantities are equivalent parameterization of the latent model, see Supplementary Information 1.1 for the details.
# 
# 2) Specify data generation parameters. Here we generate two trials of duration `100` seconds with the temporal resolution for latent trajectories `deltaT = 0.0001`. 
# 
# 3) Perform data generation, and split the generated data into training and validation datasets

# In[2]:


EnergyModelParams={'pde_solve_param':{'method':{'name': 'SEM', 'gridsize': {'Np': 8, 'Ne': 256}}}, 
               'Nv': 64, 
               'peq_model':{"model": "double_well", "params": {"xmin": 0.6, "xmax": 0.0, "depth": 2}},
               'D0': 10, 
               'num_neuron':1,
               'firing_model':[{"model": "rectified_linear", "params": {"r_slope": 100, "x_thresh": -1}}],
               'verbose':True
               }
 
data_gen_params={'deltaT':0.0001, 'time_epoch':  [(0,100)]*2}

#Initialise ground-truth em class
em_gt=neuralflow.EnergyModel(**EnergyModelParams)

#Save the ground-truth for visulalization
peq_gt=np.copy(em_gt.peq_)

#Generate data
data, time_bins, diff_traj=em_gt.generate_data(**data_gen_params)

#Split the data into training and validation set
dataTR=data[[0]]
dataCV=data[[1]]


# ## Step 2: Perform model optimization
# 1) Create another instance of `EnergyModel` that will be used for model fitting. This instance is the same as the ground-truth but with different equilibrium probability distribution `peq`. In this case, `peq` serves as the initial guess.
# 
# 2) Define fitting parameters. Here we use Gradient descent optimizer (`GD`), limit the optimization to 50 iteration, set the learning rate to `0.005`, `loglik_tol` to zero (so that optimization will not terminate due to small changes of relative loglikelihood), `etaf` to zero (so that there is no cost function regularizer), and specify validation data.
# 
# 3) Perform fitting

# In[3]:


EnergyModelParams={'pde_solve_param':{'method':{'name': 'SEM', 'gridsize': {'Np': 8, 'Ne': 256}}}, 
               'Nv': 64, 
               'peq_model':{"model": "cos_square", "params": {}},
               'D0': 10, 
               'num_neuron':1,
               'firing_model':[{"model": "rectified_linear", "params": {"r_slope": 100, "x_thresh": -1}}],
               'verbose':True
               }
em_fitting=neuralflow.EnergyModel(**EnergyModelParams)

fitting_params={'optimizer':'GD', 
               'options':{'max_iteration': 50, 'gamma': {'F': 0.005},  'loglik_tol':0.0, 'etaf': 0, 'dataCV': dataCV}}
    
em_fitting.fit(dataTR,**fitting_params)


# ## Step 3: Visualize the results

# In[4]:


lls=em_fitting.iterations_GD_['logliks']
lls_CV=em_fitting.iterations_GD_['logliksCV']

#Shift training and validated loglikelihoods such that they both start from 0 for visualisation purposes
lls= (lls-lls[0])
lls_CV=(lls_CV-lls_CV[0])

fig=plt.figure(figsize=(20,7))
gridspec.GridSpec(2,4)
Iterations=[1,6,13,50]
colors=[[0.0, 0.0, 1.0],
        [0.2, 0.4, 0.8],
        [0.4, 0.2, 0.6],
        [0.6, 0.2, 0.4]]

# Plot negative loglikelihood vs. iteration number
plt.subplot2grid((2,4), (0,0), colspan=2, rowspan=2)

plt.plot(np.arange(1,lls.size+1),lls,color='black',linewidth=3,label='training')
plt.plot(np.arange(1,lls_CV.size+1),lls_CV,color='red',linewidth=3, label='validation')
plt.xlabel('Iteration #', fontsize=18)
plt.ylabel(r'$-\log\mathcal{L}$', fontsize=18)
plt.legend()

#Point at iteration with minCV
minCV, ysize=np.argmin(lls_CV)+1, -np.min(lls_CV)/5
plt.arrow(minCV,lls_CV[minCV-1]+ysize,0,-ysize*0.9,width=0.25,length_includes_head=True,head_width=1.5,head_length=10, color='red')

#Plot potentials. Potential is calculated from peq by taking negative log: Phi = - log(peq). 
for i,Iter in enumerate(Iterations):
    plt.subplot2grid((2,4), (i//2,2+i%2))
    plt.plot(em_fitting.x_d_,np.minimum(-np.log(em_fitting.iterations_GD_['peqs'][...,Iterations[i]-1]),6),color=colors[i],linewidth=3)
    plt.plot(em_fitting.x_d_,np.minimum(-np.log(peq_gt),6),color='grey',linewidth=2)
    plt.title('Iteration {}'.format(Iterations[i])) 


# The code above should produce the following image:
# ![Jupyter notebook icon](Example1.png)
