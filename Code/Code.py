#Importing relavent libraries.
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats


# Algorithm: Metropolis, Harmonic oscillator

#Parameters for the simulation.
K = 20                          #Number of particles.
n_burn = 1000                   #Number of burn in steps.
n_prod = 100000                 #Number of production steps.
deltas = [0.01, 0.5,1.2]   # different proposal widths
colors = ['blue', 'green', 'purple']
kB = 1                          #Boltzman constant(set to 1).
T = 1                           #Tempreature of the system.
beta = 1 / (kB * T)             #Inverse boltzman constant.
k = 10.0                        #Force constant.
block_size = 10                #The size of blocks used for block averaging.

#Producing an array for the potential energies across simulation.
def Monte_Carlo(K, n_burn, n_prod, deltax, kBT=1.0, k=10.0):
  total = n_burn + n_prod       #Total number of steps.
  accepted = 0                  #Number of accepted moves.
  def V(q):                     #The total P.E for the system with configuraiton q.
    return 0.5*k*np.sum(q*q)

  q = np.zeros(K)               #Initalzied configuration space to 0.
  U = V(q)                      #Initial potential energy.

  U_samples = np.zeros(n_prod)

  for i in range(total):
    q_prop = q + (np.random.random(K) - 0.5) * deltax #Propose a new configuration.
    U_prop = V(q_prop)                                #Assosiated Proposed P.E.
    dU = U_prop - U                                   #Difference in new and previous P.E.

    X = np.random.uniform(0,1)                        #Sample from U(0,1).
    if dU <= 0.0 or np.exp(-dU / kBT) > X:            #Condition to accept move.
      q, U = q_prop, U_prop                           #Update Configuration and P.E.
      accepted += 1

    if i >= n_burn:                                   #Ensuring we sample after burn out period.
      U_samples[i - n_burn] = U


  rate = accepted / total                           #Rate at which moves are accepted.
  return U_samples, rate


def blocking(x, block_size):                          #Block averaging: returns mean and stderr of the mean.
  x = np.asarray(x)
  n_blocks = len(x) // block_size                     #Number of blocks.
  if n_blocks < 2:
    return np.mean(x), np.nan                         #Not enough blocks for an error estimate.

  blocks_mean = np.zeros(n_blocks)                    #Mean of each block.

  for i in range(n_blocks):
    start = i * block_size
    end = start + block_size
    block = x[start:end]
    blocks_mean[i] = block.mean()

  mean = np.mean(blocks_mean)                         #Mean of block means.
  std  = np.std(blocks_mean, ddof=1)                  #Std of block means.
  err  = std / np.sqrt(n_blocks)                      #Standard error of the mean.

  return mean, err

results = {}                                        #Dictionary for storing results.

#Plotting relavent simulated values.
plt.figure(figsize=(7.0, 2.6))                                  #Adjusting figure size.

for idx in range(len(deltas)):

    deltax = deltas[idx]
    color  = colors[idx]

    # Run Metropolis for deltax
    U_samples, rate = Monte_Carlo(K, n_burn, n_prod, deltax, kBT=kB, k=k)
    results[deltax] = (U_samples, rate)
    # Arrays for running estimates
    step = 2000
    Ns = np.arange(2*block_size, n_prod + 1, step)
    means = np.zeros(len(Ns))
    errs  = np.zeros(len(Ns))

    # Fill arrays using blocking
    for j in range(len(Ns)):
        N = Ns[j]
        means[j], errs[j] = blocking(U_samples[:N], block_size)

    # Skip first point (like your original code)
    Ns_plot    = Ns[1:]
    means_plot = means[1:]
    errs_plot  = errs[1:]

    # Plot this deltax
    plt.errorbar(
        Ns_plot, means_plot, yerr=errs_plot,
        fmt='-', capsize=1,
        color=color,
        ecolor='black',
        elinewidth=1,
        markeredgecolor=color,
        label=rf'Metropolis $\delta ={deltax}$'
    )
# Theoretical mean.
plt.axhline(
    0.5 * K * kB * T,
    color='red',
    linestyle='--',
    linewidth=1.5,
    label=r'Predicted $\langle U\rangle$ = $10$'
)
#Labels
plt.xlabel(r'Number of Production Samples, $\tau$')
plt.ylabel(r'Mean Potential Energy $\langle U \rangle$')
plt.legend()

#Printing relavent values
for d in deltas:

    U_samples, acc_rate = results[d]
    U_mean_M, U_err_M = blocking(U_samples, block_size)

    print(f"Delta = {d:0.2f}   <U> = {U_mean_M:.3f} ± {U_err_M:.3f}   Acceptance rate = {acc_rate:.3f}")


# Algorithm: Direct Sampling, Harmonic oscillator


K= 20                                              #K was varied to produce different figures.
#Array from direct sampling from the known distribution.
def Direct_Sampling(K, n_prod, kBT=1.0, k=10.0):
  sigma = np.sqrt(kBT / k)
  q = np.random.normal(0.0, sigma, size=(n_prod, K))
  U_samples = 0.5 * k * np.sum(q*q, axis=1)
  return U_samples

U_samples = Direct_Sampling(K, n_prod, kBT=kB*T, k=k)#Generating the P.E samples for direct importance.

step = 2000                                         #Number of steps between recording.
Ns = np.arange(200, n_prod + 1, step)               #Array for recorded points.
means = np.zeros(len(Ns))                           #Mean array for recorded points.
errs  = np.zeros(len(Ns))                           #Error array for recorded points.

for j in range(len(Ns)):                            #Filling required arrays.
    N = Ns[j]
    x = U_samples[:N]
    means[j] = np.mean(x)
    errs[j]  = np.std(x, ddof=1) / np.sqrt(N)

start_idx = 1                                       #Start at the second recorded point.

Ns    = Ns[start_idx:]
means = means[start_idx:]
errs  = errs[start_idx:]
plt.figure(figsize=(7.0, 2.6))                                  #Adjusting figure size.

#Plotting relavent simulated values.
plt.errorbar(
    Ns, means, yerr=errs,
    fmt='-', capsize=1,
    color='green',
    ecolor='black',
    elinewidth=1,
    markeredgecolor='green',
    label=r'Importance Sampling Simulated $\langle U\rangle$'
)




# Theoretical mean
plt.axhline(
    0.5 * K * kB * T,
    color='red',
    linestyle='--',
    linewidth=1.5,
    label=r'Predicted $\langle U\rangle$ = $10$'
)

#Labels
plt.xlabel(r'Number of Samples, $\tau$')
plt.ylabel(r'Mean Potential Energy $\langle U \rangle$')
plt.legend()



U_mean_D = np.mean(U_samples)                         #Extracting Final means and errors.
U_err_D  = np.std(U_samples, ddof=1) / np.sqrt(n_prod)

print(f"<U> = {U_mean_D} ± {U_err_D}")                #Printing final means +- errors.

