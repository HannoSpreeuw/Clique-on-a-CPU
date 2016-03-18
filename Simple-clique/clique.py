import numpy as np
import time

# Make a list of items, with each item four bytes.
N = 1500
# This is an array, indicating all pairs of hits which are correlated.
# 1 for correlated, 0 for uncorrelated.
# The idea is to purge it until Nassociated = Nhits
hits_method1 = np.identity(N, 'int')

# x,y or z = 1 would mean something like 1 km.

x = np.random.rand(N)
y = np.random.rand(N)
z = np.random.rand(N)

ct = np.random.rand(N)

start = time.clock()

for i in range(hits_method1.shape[0]):
    for j in range(i+1, hits_method1.shape[1]):
        # number_of_pairs_investigated += 1
        if (ct[i]-ct[j])**2 < (x[i]-x[j])**2  + (y[i] - y[j])**2 + (z[i] - z[j])**2:
            # correlated_pairs += 1
            hits_method1[i, j] = 1
            # if hit i is correlated to hit j than also vice versa.
            hits_method1[j, i] = 1

end = time.clock()

time_finding = end - start

print()
print('Time taken for finding all correlated hits with method 1 = {0}'.format(time_finding))

hits_method2 = np.identity(N, 'int')
                 
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

hits_method2[ diff < 0. ] = 1

end = time.clock()

time_finding = end - start

print()
print('Time taken for finding all correlated hits with method 2 = {0}'.format(time_finding))

# Check if the results are the same.
print()
print('(np.abs(hits_method1 - hits_method2)).sum()= {0}'.format((np.abs(hits_method1 - hits_method2)).sum()))
       
hits = hits_method2            

# Check if we still have a symmetrical matrix.    
print()
print('(hits - hits.T).sum() = {0}'.format((hits - hits.T).sum()))        
 
start = time.clock() 
            
while hits.sum() < hits.shape[0] * hits.shape[1]:
    total_hits = np.sum(hits, 0)  
    # print()
    # print(total_hits)   
    minimal_hits_index = np.argmin(total_hits)      
    hits = np.delete(hits, minimal_hits_index, 1)
    hits = np.delete(hits, minimal_hits_index, 0)
    # print()
    # print(hits.shape)
    
end = time.clock()
    
time_purging = end - start

print()
print('Time taken for purging = {0}'.format(time_purging))  
print()
print('Ratio of the time to find all correlated hits over the time for purging in order to find the maximal clique = {0}'\
      .format(time_finding/time_purging))
   
print()
print('Shape of hits is {0}'.format(hits.shape))  
print()
print('maximal clique = {0}'.format(hits.sum()))  
     
# print()
# print('Number of pairs investigated equals {0:e}'.format(number_of_pairs_investigated))
# print()
# print('Total number of pairs equals {0:e}'.format(N*(N-1)/2))
# print()
# print('Ratio of number of pairs investigated to total number of pairs equals {0:e}'.format(number_of_pairs_investigated/(N*(N-1)/2)))
# print()
# print()
# print('Number of correlated pairs equals {0:e}'.format(correlated_pairs))            
#             
