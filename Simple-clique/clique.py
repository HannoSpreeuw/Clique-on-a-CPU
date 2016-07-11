import numpy as np
import time
from numba import jit

# Make a list of items, with each item four bytes.
N = 1500

# This is an array, indicating all pairs of correlations which are correlated.
# 1 for correlated, 0 for uncorrelated.
# The idea is to purge it until Nassociated = Ncorrelations
correlations_method1 = np.identity(N, 'int')
# correlations_method1 = np.zeros((N, N), 'int')

try:
    x = np.load("x.npy")
    y = np.load("y.npy")
    z = np.load("z.npy")
    ct = np.load("ct.npy")

    assert x.size == N
    
except (FileNotFoundError, AssertionError):
    # x,y or z = 1 would mean something like 1 km.
    km = 1e3
    
    x = np.random.rand(N) * km
    y = np.random.rand(N) * km
    z = np.random.rand(N) * km
    
    # x = np.linspace(0, km, N)
    # y = np.linspace(0, km, N)
    # z = np.linspace(0, km, N)
    
    c = 3e8
    
    # ct = np.random.rand(N) * 0.1 * c
    # Divide by some number - 1e4 in this case - to get more correlated hits.
    ct = np.linspace(0, 0.1, N) * c/1e4

    np.save("x.npy", x)
    np.save("y.npy", y)
    np.save("z.npy", z)
    np.save("ct.npy", ct)
    
@jit
def compute_correlations(corr, x, y, z, ct):
    for i in range(corr.shape[0]):
        for j in range(i + 1, corr.shape[1]):
            if (ct[i]-ct[j])**2 < (x[i]-x[j])**2  + (y[i] - y[j])**2 + (z[i] - z[j])**2:
                corr[i, j] = 1
                corr[j, i] = 1


start = time.clock()

compute_correlations(correlations_method1, x, y, z, ct)

end = time.clock()

time_finding = end - start

print('Percentage correlations = {0} %'.format(100 * (np.sum(correlations_method1))/correlations_method1.size))

print()
print('Time taken for finding all correlated correlations with method 1 = {0}'.format(time_finding))

# correlations_method2 = np.identity(N, 'int')
correlations_method2 = np.zeros((N, N), 'int')

# Try to do the same thing without loops.
start = time.clock() 

combined_x = np.broadcast(x[:,np.newaxis], x)
xdiff_sq = np.empty(combined_x.shape)
xdiff_sq.flat = [(u-v)**2 for (u, v) in combined_x]

combined_y = np.broadcast(y[:,np.newaxis], y)
ydiff_sq = np.empty(combined_y.shape)
ydiff_sq.flat = [(u-v)**2 for (u, v) in combined_y]

combined_z = np.broadcast(z[:,np.newaxis], z)
zdiff_sq = np.empty(combined_z.shape)
zdiff_sq.flat = [(u-v)**2 for (u, v) in combined_z]

combined_ct = np.broadcast(ct[:,np.newaxis], ct)
ctdiff_sq = np.empty(combined_ct.shape)
ctdiff_sq.flat = [(u-v)**2 for (u, v) in combined_ct]
       
diff = ctdiff_sq - xdiff_sq - ydiff_sq - zdiff_sq

correlations_method2[ diff < 0. ] = 1

end = time.clock()

time_finding = end - start

print()
print('Time taken for finding all correlated correlations with method 2 = {0}'.format(time_finding))

# Check if the results are the same.
print()
print('(np.abs(correlations_method1 - correlations_method2)).sum()= {0}'.format((np.abs(correlations_method1 - correlations_method2)).sum()))
       
correlations = correlations_method1            

# Check if we still have a symmetrical matrix.    
print()
print('(correlations - correlations.T).sum() = {0}'.format((correlations - correlations.T).sum()))

def find_maximal_clique(corr):          

    total_corr = np.ma.array(np.sum(corr, 0), mask = False)
    corr_sum   = np.sum(total_corr)
    
    # corr_reduced_side should give the side of corr when hits with the least number of correlated hits are removed.
    # We start with the side of the initial correlations array.
    corr_reduced_side  = corr.shape[0]
    
    # Instead of removing rows and columns, we will collect them. This will enable bookkeeping of the indices of the 
    # hits in the maximal clique.
    collected_rows_and_columns = np.empty(0, 'int')
    
    while corr_sum < corr_reduced_side**2:        
        
        minsum = total_corr.min()
                  
        # These are the indices of the hits that are to be "removed".
        # Well, we will not actually remove them, because this will hamper bookkeeping.
        minimal_corr_indices = np.where(total_corr == minsum )
                
        collected_rows_and_columns = np.concatenate((collected_rows_and_columns, minimal_corr_indices[0]))
        
        number_rowcols_removed = len(minimal_corr_indices[0])        
        
        total_corr.mask[minimal_corr_indices]  =  True
        # We have to do this because not only columns, but also rows are removed.        
        total_corr                            -=  np.sum(corr[minimal_corr_indices[0], :],0)
        
        corr_reduced_side                     -=  number_rowcols_removed       
        
        corr_sum                               =  np.sum(total_corr)
        

    return collected_rows_and_columns
 
start = time.clock() 
            
rows_and_columns_to_be_masked = find_maximal_clique(correlations)

end = time.clock()

print()
print('rows_and_columns_to_be_masked = ', rows_and_columns_to_be_masked, len(rows_and_columns_to_be_masked))
# Adding a mask to the correlations and, in finding the maximal clique, modifying that mask 
# instead of the actual data enables bookkeeping.
corr_masked = np.ma.array(correlations, mask = False)
corr_masked.mask[rows_and_columns_to_be_masked, :] = True
corr_masked.mask[:, rows_and_columns_to_be_masked] = True        

clique = np.delete(correlations, rows_and_columns_to_be_masked, 0)
clique = np.delete(clique, rows_and_columns_to_be_masked, 1)

    
time_purging = end - start

print()
print('Time taken for purging = {0}'.format(time_purging))  
print()
print('Ratio of the time to find all correlated correlations over the time for purging in order to find the maximal clique = {0}'\
      .format(time_finding/time_purging))
   
print()
print('Shape of clique is {0}'.format(clique.shape))  
print()
print('maximal clique = {0}'.format(clique.sum()))  
     

