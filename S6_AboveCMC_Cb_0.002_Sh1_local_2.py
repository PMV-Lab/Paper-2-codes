# -*- coding: utf-8 -*-
"""
Created on Mon May 22 13:10:59 2023

@author: USER
"""
import numpy as np
import math
import pandas as pd
import networkx as nx
import time
import itertools
from itertools import product
import csv
import sys
import heapq
start_time = time.time()


Rgas=8.314

#K=1.5e3
#Cb=0.001
#km=[1e-6,2e-6,3e-6,4e-6,5e-6]
#Nu=25.0

#J=[[1, 3, 7, 9, 10]]


Folder_name='48_1012_nodes_var_10.0_Sh_local_S6_Cb_0.002_Gamma_14'
#Folder1='Diff IFT'
Drive ='D'  #D for server, C for desktop
np.set_printoptions(threshold=np.inf)




CI_all = []
arJ=[]
Pe_local = []  # Initialize an empty list before the loop
Sh_local = []
sigma0=0.021
R=8.314


##################################################################################
#To be altered for every surfactant


K=2.6e1
Cb0=2e-3#2x cmc=4 mM
DiffCoeff=1e-10
mol_wt=230.3 #g/mol
Temp=295
gamma_max=2.4e-5
Surf='S6'

##################################################################################

h=10e-6
n12=1.0
sigma_all=[]
alpha_all=[]
Pr_cap=[]
alpha0=1.919

beta=1.0
threshold_sigma=0.005
threshold_alpha=1.6
Pout=1e-100
tol=1e-12
tol1=1e-12
L=500e-6  #length of pt in m
mu = 8.9e-4
r_avg=50e-6
const=1
power=1/3
power=np.round(power,3)
precision=15
rho=1.0   #g/ml

#J=[]

sigma_t=[]
alpha_t=[]
Cb_max_t=[]
Ci_t=[]
Cff_t=[]
Csf_t=[]
km_t=[]
Cond_t=[]
A2=[]
A3=[]

N=22
n2=10.0
n3=5.0e-06
Nodes=2*N*(N+1)
sigma=sigma0*np.ones((2*N*(N+1), 1))
File=1
Nature='GAMMA'


# Ensure n3 is formatted to appear in scientific notation with lowercase 'e'
formatted_n3 = f"{n3:.1e}"

# Create the file path using an f-string with detailed control over formatting
df_path = rf"{Drive}:\Users\debanikb\OneDrive - Technion\Research_Technion\Python_PNM\Surfactant A-D\{Folder_name}\{File}\Inv_Radius_{Nodes}_{n2}_{formatted_n3}{Nature}.csv"

# Load the dataframe from CSV
df = pd.read_csv(df_path, header=None)
inv_r = df.to_numpy()

# inv_r=np.array([[19371.82250346, 20264.94910211, 19697.49114878, 20151.47031158,
#         19899.73292203, 19912.05009437, 19567.94111435, 20777.21170365,
#         19525.03386252, 20375.7252995 , 20397.43278826, 19371.82250346]])





# A=np.array([[1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0],
#         [0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0],
#         [1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0],
#         [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
#         [0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0],
#         [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0],
#         [0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1],
#         [0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0],
#         [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
#         [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1],
#         [0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0],
#         [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1]])



T=-2*(inv_r)*np.cos(alpha0)-2*(1/h)*np.cos(alpha0)

def get_neighbor_indices(position, dimensions):
    '''
    dimensions is a shape of np.array
    '''
    i, j = position
    indices = [(i+1,j), (i-1,j), (i,j+1), (i,j-1)]#[(i+1,j),(i,j+1),(i+1,j+1),(i-1,j-1),(i-1,j),(i,j-1),(i+1,j-1),(i-1,j+1)]  # [(i+1,j), (i-1,j), (i,j+1), (i,j-1)]
    return [
        (i,j) for i,j in indices
        if i>=0 and i<dimensions[0]
            and j>=0 and j<dimensions[1]
        ]

def iterate_array(init_i, init_j, arr, condition_func):
    '''
    arr is an instance of np.array
    condition_func is a function (value) => boolean
    '''
    indices_to_check = [(init_i,init_j)]
    checked_indices = set()
    result = []
    t0 = None
    t1 = None
    timestamps = []
    while indices_to_check:
        pos = indices_to_check.pop()
        if pos in checked_indices:
            continue
        item = arr[pos]
        checked_indices.add(pos)
        if condition_func(item):
            result.append(item)
            t1=time.time()
            if(t0==None):
                t0=t1
            timestamps.append(t1-t0)
            indices_to_check.extend(
                get_neighbor_indices(pos, arr.shape)
            )
    return result,timestamps


def invasion(N):

    def pos():
        x, y = 1, N + 3 - 1
        for _ in range(2 * N * (N + 1)):
            yield (x, y)
            y -= (x + 2) // (N + 3)
            x = (x + 2) % (N + 3)

    G = nx.Graph()
    it_pos = pos()
    for u in range(2 * N * (N + 1)):
        G.add_node(u+1, pos=next(it_pos))
        if u % (2 * N + 1) < N:
            for v in (u - 2 * N - 1, u - N - 1, u - N):
                if G.has_node(v + 1):
                    G.add_edge(u + 1, v + 1)
        elif u % (2 * N + 1) == N:
            G.add_edge(u + 1, u - N + 1)
        elif u % (2 * N + 1) < 2 * N:
            for v in (u - 1, u - N - 1, u - N):
                G.add_edge(u + 1, v + 1)
        else:
            for v in (u - 1, u - N - 1):
                G.add_edge(u + 1, v + 1)

    nx.draw(G, nx.get_node_attributes(G, 'pos'), with_labels=True, font_weight='bold')

#toc()

#tic()
    Nodes=len(G.nodes)
#toc()
#print("Nodes =",Nodes)

#tic()
    A=nx.adjacency_matrix(G).todense()
#toc()

#tic()
    A = np.squeeze(np.asarray(A))
    #print("A here",[A])
#toc()

#toc()
#print("A before adding 1 on diagonal =",[A])

#tic()
    np.fill_diagonal(A, 1)        

    return A

def Entry_Pressure(A):

    # print("sigma here =",sigma)
    # print("T here =",T)
    
    
    Test=sigma*T

    New=np.diag(Test)

    Pe=A*New

    Distinct=sorted(set(list(np.reshape(Pe,(2*N*(N+1))**2))))

    Distinct = [x for x in Distinct if x!=0]

    Indices = [[] for x in Distinct]
    # V=np.average(Distinct)
    # print("V here =",V)
    Pin0=1.000*np.average(Distinct) #np.average(Distinct)   #1.0005 for rect
    #print("Pin0 here =",Pin0)
    #print("len(J[0])/Nodes =",len(J[0])/Nodes)

    Visited_Elements,timestamps=iterate_array(0,0, Pe, lambda x : x < Pin0)

    Visited_Indices = list(zip(*np.where(np.isin(Pe, Visited_Elements))))

    Anew = [a for a in Visited_Elements if a!=0.0]

    Bnew = [b for a,b in zip(Visited_Elements,Visited_Indices) if a!=0.0]
    Anew = [Anew] # probably unnecesary
    Bnew = [Bnew]

    Total_Indices=list(product(*map(range, Pe.shape)))

    Not_Visited_Indices = set(Total_Indices)-set(Visited_Indices)
    Not_Visited_Indices = list(Not_Visited_Indices)

    J=list({j for i,j in Not_Visited_Indices})
    J.sort()
    J=[[i for i in J]]
    #print("J here =", J)

    Pe = np.delete(Pe, J)
    
    
    return J,Pin0,Total_Indices,New #,Nodes_Unvisited

def Entry_Pressure1(A):

    # print("sigma here =",sigma)
    # print("T here =",T)
    
    
    Test=sigma*T

    New=np.diag(Test)

    Pe=A*New

    Distinct=sorted(set(list(np.reshape(Pe,(2*N*(N+1))**2))))

    Distinct = [x for x in Distinct if x!=0]

    Indices = [[] for x in Distinct]
    # V=np.average(Distinct)
    # print("V here =",V)
    #Pin0=1.000*np.average(Distinct) #np.average(Distinct)   #1.0005 for rect
    #print("Pin0 here =",Pin0)
    #print("len(J[0])/Nodes =",len(J[0])/Nodes)

    Visited_Elements,timestamps=iterate_array(0,0, Pe, lambda x : x < Pin0)

    Visited_Indices = list(zip(*np.where(np.isin(Pe, Visited_Elements))))

    Anew = [a for a in Visited_Elements if a!=0.0]

    Bnew = [b for a,b in zip(Visited_Elements,Visited_Indices) if a!=0.0]
    Anew = [Anew] # probably unnecesary
    Bnew = [Bnew]

    Total_Indices=list(product(*map(range, Pe.shape)))

    Not_Visited_Indices = set(Total_Indices)-set(Visited_Indices)
    Not_Visited_Indices = list(Not_Visited_Indices)

    J=list({j for i,j in Not_Visited_Indices})
    J.sort()
    J=[[i for i in J]]
    #print("J here =", J)

    Pe = np.delete(Pe, J)
    
    
    return J,Pin0,Total_Indices #,Nodes_Unvisited

def Change_Adjacency(A):
    
    mask = np.ones(A.shape[0], bool)
    mask[[[i for i in J[0] if i!=0]]] = 0
    A1=A[mask,:][:,mask]
    np.fill_diagonal(A1, 0)

    return A1


def Change_Adjacency1(A):
    
    mask = np.ones(A.shape[0], bool)
    mask[[[i for i in J1[0] if i!=0]]] = 0
    A1=A[mask,:][:,mask]
    np.fill_diagonal(A1, 0)

    return A1

def Change_Adjacency2(A):
    
    mask = np.ones(A.shape[0], bool)
    mask[[[i for i in J2[0] if i!=0]]] = 0
    A1=A[mask,:][:,mask]
    np.fill_diagonal(A1, 0)

    return A1

def Change_Adjacency(A):
    
    mask = np.ones(A.shape[0], bool)
    mask[[[i for i in J[0] if i!=0]]] = 0
    A1=A[mask,:][:,mask]
    np.fill_diagonal(A1, 0)

    return A1


def pressure_flux_velocity1():
        
           
        
        #print("inv_r at t=0 ",[inv_r1])

        R=(12*mu*L)*(1/(h**3))*(inv_rJ1)*(1/(1-0.63*h*inv_rJ1))

        #R=8*(1/np.pi)*L*mu*inv_r1**4   #Total Resistance in Pa-s/m^3
        #print("Total Resistance, R =",[R])

        # ######################################################################################


        R1=R*A1
        #print("R1 =",[R1])
        #print("Modified Resistance Adj. matrix, R2 =",[R1]) #Modified Resistance*Adjacency matrix of connected path

        R2=np.divide(1, R1, where=R1!=0, out=R1).sum(1) #[1/R2 summation of each non-zero element of each row ]
        #print("1/R2 summation of each non-zero element of each row, R2 =",[R2])

        B=1/R1
        #print("B =",[B])

        R3 = np.where(np.isfinite(B), B, 0)
        #print("R3 =",[R3])


        ######################################################################################

        R3_mod=R3[1:-1,1:-1] #Deleting first and last rows and columns 
        #print([R3_mod])

        ######################################################################################


        R3_mod[R3_mod != 0] = 1 / R3_mod[R3_mod != 0]
        #print("R3_mod =",[R3_mod])

        D=R2[1:len(A)-1] #Selecting the counts on rows 1-9 for 10x10
        #print("D =",[D])

        ######################################################################################

        np.fill_diagonal(R3_mod, -D)


        ######################################################################################

        Hnew=R3[:,0] #Selecting first column 
        # Hnew=R[0,:] #Selecting first row
        #print("Hnew =",[Hnew])
        Hnew2=Hnew[1:-1] #Deleting first and last value of the column
        P=-Pin0/Hnew2
        P2 = np.where(np.isfinite(P), P, 0)
        #print("P =",[P2])

        ######################################################################################

        # P_sps = sps.coo_matrix(R3_mod)
        # X = spsl.inv(P_sps).dot(P2)    #pinv
        #X = scipy.linalg.inv(-R3_mod).dot(-P2)
        #X = np.linalg.pinv(R3_mod).dot(P2)
        #X=solve(R3_mod,P2)
        #tic()
        # rev_array = R3_mod[::-1]
        # X = np.array(rev_array).dot(-P2)
        #X = np.linalg.pinv(R3_mod).dot(P2)
        
        #X, residuals, rank, s = np.linalg.lstsq(R3_mod, P2, rcond=None)
        # print("R3 mod=",[R3_mod])
        # print("P2 =",[P2])
        X = np.linalg.inv(-R3_mod).dot(-P2)
        #print("X =",X)
        # R3_mod = csc_matrix(R3_mod)
        # P2 = csc_matrix(P2)
        # X, info = cg(R3_mod, P2)
        #X=spsolve(R3_mod,P2)
        #toc()
        # X=np.linalg.lstsq(R3_mod,P2)
        #X= scipy.linalg.inv(R3_mod).dot(P2)
        #X=R3_mod.I
        
        #R3_mod= csc_matrix(R3_mod)

        #b = np.array([-1, -0.5, -1, 2])

        #X, exit_code = cg(R3_mod, P2)

        
        # P_sps = sps.coo_matrix(R3_mod)
        # X = spsl.inv(P_sps).dot(P2)    #pinv
        # #X = np.linalg.pinv(R3_mod).dot(P2)
        # #tic()
        # #X = scipy.linalg.pinv(R3_mod).dot(P2)
        # #toc()
        # # X = spsolve(P_sps,P2)
        #X = scipy.linalg.solve(R3_mod,P2)
        # X = spsolve(P_sps,P2)
        #X = np.linalg.solve(R3_mod,P2)
        # a_sps = scipy.sparse.csc_matrix(a) 
        # lu_obj = scipy.sparse.linalg.splu(a_sps)
        #print("X =",[X])

              

#         ##################FLUX CALCULATION##########################

        X = np.insert(X, 0, Pin0) #Placing inlet pressure (10) at the start
        #print("min X value=",[min(X)])
        X = np.insert(X, len(X), Pout) #Placing outlet pressure (0) at the end
        X1=np.unique(X)
        #print("X shape =",[X.shape])
        #print("X1 unique values shape =",[X1.shape])
        #Nodes_Unvisited=(len(J)/math.sqrt(len(Total_Indices))*100)
        #print("% nodes not visited =",Nodes_Unvisited)
        #print("Pressure drop, X =",[X]) #Complete pressure matrix


#         ######################################################################################

        np.fill_diagonal(A1, 0)

        Result = -A1*X #Multiplying first element of X to all the elements of first column of A and so on
        #print("Result =",[Result])
#       #################################################################

        # Step 3: Matrix operation 

        out = np.where(Result!=0, X[:, None] + Result, Result) #Adding each element of X to non-zero elements row-wise
        #print("out =",[out]) 

#       #################################################################

#       # Step 4: Flux between nodes

        Flux=out/B  #Flux matrix
        #print("Flux =",[Flux]) 
        
        Sum=np.sum(Flux,axis=1)
        #print("Sum =",[Sum])

        Sum[np.abs(Sum) < tol] = 0.0
        #print("Sum =",Sum)

        #################################################################

        # Step 5: Extracting postive elements with their indices 

        values = Flux[Flux > 0,None]
        #print("Positive values =",[values])

        indices = np.array(np.where(Flux > 0)).T
        #print("Indices =",[indices])


#         ######################################################################################
        
#         ##################VELOCITY CALCULATION##########################

        T1=inv_rJ1.flatten()
        #print([T1])

        invr_adj=T1*A1
        #print("Inverse of radius adjacency matrix of the connected path, r_adj =",[invr_adj])

#         #######################################################################################

#         #2 Inverse of radius values corresponding to indices

        invr_indices = np.array([[invr_adj[i0][i1]] for i0, i1 in indices])
        #print("Inverse of radius values corresponding to indices, invr_indices=",[invr_indices])

        #######################################################################################

        #3  Velocity Profiles

        # Velprof=(2/np.pi)*values*(((invr_indices)**2-(r**2)*(invr_indices)**4))
        # print("Velocity Profiles, Velprof=", [Velprof])
        
        Velprof = values*invr_indices/(h)
        
        return X,Flux,Velprof,indices
    
    
def pressure_flux_velocity():
        
           
        
        #print("inv_r at t=0 ",[inv_r1])

        R=(12*mu*L)*(1/(h**3))*(inv_rJ1)*(1/(1-0.63*h*inv_rJ1))

        #R=8*(1/np.pi)*L*mu*inv_r1**4   #Total Resistance in Pa-s/m^3
        #print("Total Resistance, R =",[R])

        # ######################################################################################


        R1=R*A1
        #print("R1 =",[R1])
        #print("Modified Resistance Adj. matrix, R2 =",[R1]) #Modified Resistance*Adjacency matrix of connected path

        R2=np.divide(1, R1, where=R1!=0, out=R1).sum(1) #[1/R2 summation of each non-zero element of each row ]
        #print("1/R2 summation of each non-zero element of each row, R2 =",[R2])

        B=1/R1
        #print("B =",[B])

        R3 = np.where(np.isfinite(B), B, 0)
        #print("R3 =",[R3])


        ######################################################################################

        R3_mod=R3[1:-1,1:-1] #Deleting first and last rows and columns 
        #print([R3_mod])

        ######################################################################################


        R3_mod[R3_mod != 0] = 1 / R3_mod[R3_mod != 0]
        #print("R3_mod =",[R3_mod])

        D=R2[1:len(A)-1] #Selecting the counts on rows 1-9 for 10x10
        #print("D =",[D])

        ######################################################################################

        np.fill_diagonal(R3_mod, -D)


        ######################################################################################

        Hnew=R3[:,0] #Selecting first column 
        # Hnew=R[0,:] #Selecting first row
        #print("Hnew =",[Hnew])
        Hnew2=Hnew[1:-1] #Deleting first and last value of the column
        P=-Pin0/Hnew2
        P2 = np.where(np.isfinite(P), P, 0)
        #print("P =",[P2])

        ######################################################################################

        # P_sps = sps.coo_matrix(R3_mod)
        # X = spsl.inv(P_sps).dot(P2)    #pinv
        #X = scipy.linalg.inv(-R3_mod).dot(-P2)
        #X = np.linalg.pinv(R3_mod).dot(P2)
        #X=solve(R3_mod,P2)
        #tic()
        # rev_array = R3_mod[::-1]
        # X = np.array(rev_array).dot(-P2)
        #X = np.linalg.pinv(R3_mod).dot(P2)
        
        #X, residuals, rank, s = np.linalg.lstsq(R3_mod, P2, rcond=None)
        # print("R3 mod=",[R3_mod])
        # print("P2 =",[P2])
        X = np.linalg.pinv(R3_mod).dot(P2)
        #print("X =",X)
        # R3_mod = csc_matrix(R3_mod)
        # P2 = csc_matrix(P2)
        # X, info = cg(R3_mod, P2)
        #X=spsolve(R3_mod,P2)
        #toc()
        # X=np.linalg.lstsq(R3_mod,P2)
        #X= scipy.linalg.inv(R3_mod).dot(P2)
        #X=R3_mod.I
        
        #R3_mod= csc_matrix(R3_mod)

        #b = np.array([-1, -0.5, -1, 2])

        #X, exit_code = cg(R3_mod, P2)

        
        # P_sps = sps.coo_matrix(R3_mod)
        # X = spsl.inv(P_sps).dot(P2)    #pinv
        # #X = np.linalg.pinv(R3_mod).dot(P2)
        # #tic()
        # #X = scipy.linalg.pinv(R3_mod).dot(P2)
        # #toc()
        # # X = spsolve(P_sps,P2)
        #X = scipy.linalg.solve(R3_mod,P2)
        # X = spsolve(P_sps,P2)
        #X = np.linalg.solve(R3_mod,P2)
        # a_sps = scipy.sparse.csc_matrix(a) 
        # lu_obj = scipy.sparse.linalg.splu(a_sps)
        #print("X =",[X])

              

#         ##################FLUX CALCULATION##########################

        X = np.insert(X, 0, Pin0) #Placing inlet pressure (10) at the start
        #print("min X value=",[min(X)])
        X = np.insert(X, len(X), Pout) #Placing outlet pressure (0) at the end
        X1=np.unique(X)
        #print("X shape =",[X.shape])
        #print("X1 unique values shape =",[X1.shape])
        #Nodes_Unvisited=(len(J)/math.sqrt(len(Total_Indices))*100)
        #print("% nodes not visited =",Nodes_Unvisited)
        #print("Pressure drop, X =",[X]) #Complete pressure matrix


#         ######################################################################################

        np.fill_diagonal(A1, 0)

        Result = -A1*X #Multiplying first element of X to all the elements of first column of A and so on
        #print("Result =",[Result])
#       #################################################################

        # Step 3: Matrix operation 

        out = np.where(Result!=0, X[:, None] + Result, Result) #Adding each element of X to non-zero elements row-wise
        #print("out =",[out]) 

#       #################################################################

#       # Step 4: Flux between nodes

        Flux=out/B  #Flux matrix
        #print("Flux =",[Flux]) 
        
        Sum=np.sum(Flux,axis=1)
        #print("Sum =",[Sum])

        Sum[np.abs(Sum) < tol] = 0.0
        #print("Sum =",Sum)

        #################################################################

        # Step 5: Extracting postive elements with their indices 

        values = Flux[Flux > 0,None]
        #print("Positive values =",[values])

        indices = np.array(np.where(Flux > 0)).T
        #print("Indices =",[indices])


#         ######################################################################################
        
#         ##################VELOCITY CALCULATION##########################

        T1=inv_rJ1.flatten()
        #print([T1])

        invr_adj=T1*A1
        #print("Inverse of radius adjacency matrix of the connected path, r_adj =",[invr_adj])

#         #######################################################################################

#         #2 Inverse of radius values corresponding to indices

        invr_indices = np.array([[invr_adj[i0][i1]] for i0, i1 in indices])
        #print("Inverse of radius values corresponding to indices, invr_indices=",[invr_indices])

        #######################################################################################

        #3  Velocity Profiles

        # Velprof=(2/np.pi)*values*(((invr_indices)**2-(r**2)*(invr_indices)**4))
        # print("Velocity Profiles, Velprof=", [Velprof])
        
        Velprof = values*invr_indices/(h)
        
        return X,Flux,Velprof,indices
    
def pressure_flux_velocity2():
        
           
        
        #print("inv_r at t=0 ",[inv_r1])

        R=(12*mu*L)*(1/(h**3))*(inv_rJ2)*(1/(1-0.63*h*inv_rJ2))

        #R=8*(1/np.pi)*L*mu*inv_r1**4   #Total Resistance in Pa-s/m^3
        #print("Total Resistance, R =",[R])

        # ######################################################################################


        R1=R*A1
        #print("R1 =",[R1])
        #print("Modified Resistance Adj. matrix, R2 =",[R1]) #Modified Resistance*Adjacency matrix of connected path

        R2=np.divide(1, R1, where=R1!=0, out=R1).sum(1) #[1/R2 summation of each non-zero element of each row ]
        #print("1/R2 summation of each non-zero element of each row, R2 =",[R2])

        B=1/R1
        #print("B =",[B])

        R3 = np.where(np.isfinite(B), B, 0)
        #print("R3 =",[R3])


        ######################################################################################

        R3_mod=R3[1:-1,1:-1] #Deleting first and last rows and columns 
        #print([R3_mod])

        ######################################################################################


        R3_mod[R3_mod != 0] = 1 / R3_mod[R3_mod != 0]
        #print("R3_mod =",[R3_mod])

        D=R2[1:len(A)-1] #Selecting the counts on rows 1-9 for 10x10
        #print("D =",[D])

        ######################################################################################

        np.fill_diagonal(R3_mod, -D)


        ######################################################################################

        Hnew=R3[:,0] #Selecting first column 
        # Hnew=R[0,:] #Selecting first row
        #print("Hnew =",[Hnew])
        Hnew2=Hnew[1:-1] #Deleting first and last value of the column
        P=-Pin0/Hnew2
        P2 = np.where(np.isfinite(P), P, 0)
        #print("P =",[P2])

        ######################################################################################

        # P_sps = sps.coo_matrix(R3_mod)
        # X = spsl.inv(P_sps).dot(P2)    #pinv
        #X = scipy.linalg.inv(-R3_mod).dot(-P2)
        #X = np.linalg.pinv(R3_mod).dot(P2)
        #X=solve(R3_mod,P2)
        #tic()
        # rev_array = R3_mod[::-1]
        # X = np.array(rev_array).dot(-P2)
        #X = np.linalg.pinv(R3_mod).dot(P2)
        
        #X, residuals, rank, s = np.linalg.lstsq(R3_mod, P2, rcond=None)
        # print("R3 mod=",[R3_mod])
        # print("P2 =",[P2])
        X = np.linalg.pinv(R3_mod).dot(P2)
        #print("X =",X)
        # R3_mod = csc_matrix(R3_mod)
        # P2 = csc_matrix(P2)
        # X, info = cg(R3_mod, P2)
        #X=spsolve(R3_mod,P2)
        #toc()
        # X=np.linalg.lstsq(R3_mod,P2)
        #X= scipy.linalg.inv(R3_mod).dot(P2)
        #X=R3_mod.I
        
        #R3_mod= csc_matrix(R3_mod)

        #b = np.array([-1, -0.5, -1, 2])

        #X, exit_code = cg(R3_mod, P2)

        
        # P_sps = sps.coo_matrix(R3_mod)
        # X = spsl.inv(P_sps).dot(P2)    #pinv
        # #X = np.linalg.pinv(R3_mod).dot(P2)
        # #tic()
        # #X = scipy.linalg.pinv(R3_mod).dot(P2)
        # #toc()
        # # X = spsolve(P_sps,P2)
        #X = scipy.linalg.solve(R3_mod,P2)
        # X = spsolve(P_sps,P2)
        #X = np.linalg.solve(R3_mod,P2)
        # a_sps = scipy.sparse.csc_matrix(a) 
        # lu_obj = scipy.sparse.linalg.splu(a_sps)
        #print("X =",[X])

              

#         ##################FLUX CALCULATION##########################

        X = np.insert(X, 0, Pin0) #Placing inlet pressure (10) at the start
        #print("min X value=",[min(X)])
        X = np.insert(X, len(X), Pout) #Placing outlet pressure (0) at the end
        X1=np.unique(X)
        #print("X shape =",[X.shape])
        #print("X1 unique values shape =",[X1.shape])
        #Nodes_Unvisited=(len(J)/math.sqrt(len(Total_Indices))*100)
        #print("% nodes not visited =",Nodes_Unvisited)
        #print("Pressure drop, X =",[X]) #Complete pressure matrix


#         ######################################################################################

        np.fill_diagonal(A1, 0)

        Result = -A1*X #Multiplying first element of X to all the elements of first column of A and so on
        #print("Result =",[Result])
#       #################################################################

        # Step 3: Matrix operation 

        out = np.where(Result!=0, X[:, None] + Result, Result) #Adding each element of X to non-zero elements row-wise
        #print("out =",[out]) 

#       #################################################################

#       # Step 4: Flux between nodes

        Flux=out/B  #Flux matrix
        #print("Flux =",[Flux]) 
        
        Sum=np.sum(Flux,axis=1)
        #print("Sum =",[Sum])

        Sum[np.abs(Sum) < tol] = 0.0
        #print("Sum =",Sum)

        #################################################################

        # Step 5: Extracting postive elements with their indices 

        values = Flux[Flux > 0,None]
        #print("Positive values =",[values])

        indices = np.array(np.where(Flux > 0)).T
        #print("Indices =",[indices])


#         ######################################################################################
        
#         ##################VELOCITY CALCULATION##########################

        T1=inv_rJ2.flatten()
        #print([T1])

        invr_adj=T1*A1
        #print("Inverse of radius adjacency matrix of the connected path, r_adj =",[invr_adj])

#         #######################################################################################

#         #2 Inverse of radius values corresponding to indices

        invr_indices = np.array([[invr_adj[i0][i1]] for i0, i1 in indices])
        #print("Inverse of radius values corresponding to indices, invr_indices=",[invr_indices])

        #######################################################################################

        #3  Velocity Profiles

        # Velprof=(2/np.pi)*values*(((invr_indices)**2-(r**2)*(invr_indices)**4))
        # print("Velocity Profiles, Velprof=", [Velprof])
        
        Velprof = values*invr_indices/(h)
        
        return X,Flux,Velprof,indices
    




def bulk_conc(Flux,J): 
    
    J=[[i for i in J[0] if i!=0]]
    
    B=Cb0*Flux[0][1:-1]

    Sum=np.where(Flux>0, Flux, 0).sum(axis=1)

    for i in range(0,len(Flux)):
        Flux[i,i]=Sum[i] 
#print([Flux])

    Flux=Flux[1:-1,1:-1]
#print([Flux])

    x, y = np.where(Flux>0)
    m = x!=y
    Flux[x[m], y[m]] = 0
#print("Modified flux =",[Flux])

    Cb = np.linalg.pinv(Flux).dot(B)
#print("Different bulk concentrations =", [C]) #excluding the last pore throat


    Cb = np.insert(Cb, 0, Cb0)
#print([C])

    for i in range(0,len(Cb)):
        if (Cb[i]<0):
            Cb[i]=0
#print([C])



    for i in range(0,len(J[0])): 
        Cb = np.insert(Cb, J[0][i], 0.0)
    #print([C])
    
    Cb=Cb.tolist()
    Cb =[[i for i in Cb]]


    return Cb




def local_pressure1(indices,Velprof,X,A):

    T2=np.unique(indices)
    #print(T2)
    T2=list(T2)
    #print("T2 =",T2)
    res = [ele for ele in range(max(T2)+1) if ele not in T2]
    #print("res =",res)
    T2=T2+res
    #print("T2 =",T2)
    T2.sort()
    #print("T2 =",T2)

##################################################################

    #J=[[i for i in J[0] if i!=0]]

    for i in range(len(J2[0])): 
        T2=T2[:1] + [i + 1 for i in T2] 
    #print("All T2=",T2)


    for i in range(0,len(J2[0])):
        T2.remove(J2[0][i])
    #print("Corr original indices =",T2)
    
    #return T2


##################################################################

#print("J 0 =",J) 

    


    def specific(T2, Z):
        tmp = Z[0][0]
        for i in range(0, len(tmp)):
            pair = tmp[i]
            pair[0] = T2[pair[0]]
            pair[1] = T2[pair[1]] 
        return Z

    Ii1=[indices[np.newaxis,:]]
    Iv1=[Velprof[np.newaxis,:]]
    
    indices=specific(T2,Ii1)
    #print("Renumbering indices =",V)

##################################################################

    def make_array(indices2, values):
        rtrn = np.zeros(np.max(indices2) + 1, dtype=values.dtype)
        rtrn[indices2] = values
        return rtrn

    Xnew=make_array(T2, X) 
    #print("New X =",[Xnew])

##################################################################


    Pr=A*Xnew
    #print("Pr =",[Pr])

    MaxPr=Pr.max(axis=1)
    #print("MaxPr =",[MaxPr])
# [array([1731.42069024, 1517.57806537, 1731.42069024, 1731.42069024,
#        1517.57806537, 1731.42069024, 1517.57806537, 1302.70214927,
#        1302.70214927,  649.78119768, 1302.70214927,  649.78119768])]
    arMaxPrnew=[]
    arindices_MaxPrnew=[]
    
    indices_MaxPr = np.argmax(Pr, axis=1) 
    indices_MaxPr=indices_MaxPr.tolist()
    #print("indices_MaxPr =",indices_MaxPr)

    for i in range(0,len(J1)): 
        MaxPrnew=MaxPr[J1[i]]
        arMaxPrnew.append(MaxPrnew)
        MaxPrnew=list(arMaxPrnew)
        
    for i in range(0,len(J1[0])):  
        indices_MaxPrnew=indices_MaxPr[J1[0][i]]
        arindices_MaxPrnew.append(indices_MaxPrnew)
        indices_MaxPrnew=list(arindices_MaxPrnew)
        
    # indices_MaxPrnew = np.argmax(MaxPrnew, axis=1) 
    # indices_MaxPrnew=indices_MaxPrnew.tolist()
    # print("indices_MaxPrnew =",indices_MaxPrnew)
        
    return MaxPrnew,indices_MaxPrnew,indices


def local_pressure(indices,Velprof,X,A):

    T2=np.unique(indices)
    #print(T2)
    T2=list(T2)
    #print("T2 =",T2)
    res = [ele for ele in range(max(T2)+1) if ele not in T2]
    #print("res =",res)
    T2=T2+res
    #print("T2 =",T2)
    T2.sort()
    #print("T2 =",T2)

##################################################################

    #J=[[i for i in J[0] if i!=0]]

    for i in range(len(J[0])): 
        T2=T2[:1] + [i + 1 for i in T2] 
    #print("All T2=",T2)


    for i in range(0,len(J[0])):
        T2.remove(J[0][i])
    #print("Corr original indices =",T2)
    
    #return T2


##################################################################

#print("J 0 =",J) 

    


    def specific(T2, Z):
        tmp = Z[0][0]
        for i in range(0, len(tmp)):
            pair = tmp[i]
            pair[0] = T2[pair[0]]
            pair[1] = T2[pair[1]] 
        return Z

    Ii1=[indices[np.newaxis,:]]
    Iv1=[Velprof[np.newaxis,:]]
    
    indices=specific(T2,Ii1)
    #print("Renumbering indices =",V)

##################################################################

    def make_array(indices2, values):
        rtrn = np.zeros(np.max(indices2) + 1, dtype=values.dtype)
        rtrn[indices2] = values
        return rtrn

    Xnew=make_array(T2, X) 
    #print("New X =",[Xnew])

##################################################################


    Pr=A*Xnew
    #print("Pr =",[Pr])

    MaxPr=Pr.max(axis=1)
    #print("MaxPr =",[MaxPr])
# [array([1731.42069024, 1517.57806537, 1731.42069024, 1731.42069024,
#        1517.57806537, 1731.42069024, 1517.57806537, 1302.70214927,
#        1302.70214927,  649.78119768, 1302.70214927,  649.78119768])]
    arMaxPrnew=[]
    arindices_MaxPrnew=[]
    
    indices_MaxPr = np.argmax(Pr, axis=1) 
    indices_MaxPr=indices_MaxPr.tolist()
    #print("indices_MaxPr =",indices_MaxPr)

    for i in range(0,len(J)): 
        MaxPrnew=MaxPr[J[i]]
        arMaxPrnew.append(MaxPrnew)
        MaxPrnew=list(arMaxPrnew)
        
    for i in range(0,len(J[0])):  
        indices_MaxPrnew=indices_MaxPr[J[0][i]]
        arindices_MaxPrnew.append(indices_MaxPrnew)
        indices_MaxPrnew=list(arindices_MaxPrnew)
        
    # indices_MaxPrnew = np.argmax(MaxPrnew, axis=1) 
    # indices_MaxPrnew=indices_MaxPrnew.tolist()
    # print("indices_MaxPrnew =",indices_MaxPrnew)
        
    return MaxPrnew,indices_MaxPrnew,indices


#print("inv_rJ1 =",inv_rJ1)

##########################################################################################################

def local_pressure2(indices,Velprof,X,A):

    T2=np.unique(indices)
    #print(T2)
    T2=list(T2)
    #print("T2 =",T2)
    res = [ele for ele in range(max(T2)+1) if ele not in T2]
    #print("res =",res)
    T2=T2+res
    #print("T2 =",T2)
    T2.sort()
    #print("T2 =",T2)

##################################################################

    #J=[[i for i in J[0] if i!=0]]

    for i in range(len(J[0])): 
        T2=T2[:1] + [i + 1 for i in T2] 
    #print("All T2=",T2)


    for i in range(0,len(J[0])):
        T2.remove(J[0][i])
    #print("Corr original indices =",T2)
    
    #return T2


##################################################################

#print("J 0 =",J) 

    


    def specific(T2, Z):
        tmp = Z[0][0]
        for i in range(0, len(tmp)):
            pair = tmp[i]
            pair[0] = T2[pair[0]]
            pair[1] = T2[pair[1]] 
        return Z

    Ii1=[indices[np.newaxis,:]]
    Iv1=[Velprof[np.newaxis,:]]
    
    indices=specific(T2,Ii1)
    #print("Renumbering indices =",V)

##################################################################

    def make_array(indices2, values):
        rtrn = np.zeros(np.max(indices2) + 1, dtype=values.dtype)
        rtrn[indices2] = values
        return rtrn

    Xnew=make_array(T2, X) 
    #print("New X =",[Xnew])

##################################################################


    Pr=A*Xnew
    #print("Pr =",[Pr])

    MaxPr=Pr.max(axis=1)
    #print("MaxPr =",[MaxPr])
# [array([1731.42069024, 1517.57806537, 1731.42069024, 1731.42069024,
#        1517.57806537, 1731.42069024, 1517.57806537, 1302.70214927,
#        1302.70214927,  649.78119768, 1302.70214927,  649.78119768])]
    arMaxPrnew=[]
    arindices_MaxPrnew=[]
    
    indices_MaxPr = np.argmax(Pr, axis=1) 
    indices_MaxPr=indices_MaxPr.tolist()
    #print("indices_MaxPr =",indices_MaxPr)

    for i in range(0,len(J)): 
        MaxPrnew=MaxPr[J[i]]
        arMaxPrnew.append(MaxPrnew)
        MaxPrnew=list(arMaxPrnew)
        
    for i in range(0,len(J[0])):  
        indices_MaxPrnew=indices_MaxPr[J[0][i]]
        arindices_MaxPrnew.append(indices_MaxPrnew)
        indices_MaxPrnew=list(arindices_MaxPrnew)
        
    # indices_MaxPrnew = np.argmax(MaxPrnew, axis=1) 
    # indices_MaxPrnew=indices_MaxPrnew.tolist()
    # print("indices_MaxPrnew =",indices_MaxPrnew)
        
    return MaxPrnew,indices_MaxPrnew,indices


######################################################################################################### 

def Cb_max1(A,J,Cb): 
    
    Cb_max=[]
    np.fill_diagonal(A, 0)

    out = [[[i], np.where(a==1)[0].tolist()] for i, a in enumerate(A)]
    #print(out) 
    
    for i in range(0,len(J[0])):
        res1=J[0][i]
        #print(res1)
        res2=out[res1][1]
        #print(res2)
        res3 = max(Cb[0][i] for i in res2)
        #print(res3)
        Cb_max.append(res3)
    #print(Cb_max)
    
    return Cb_max


######################################################################################################### 



# Assuming New is an array obtained from your code
A = invasion(N)
J, Pin0, Total_Indices, New = Entry_Pressure(A)
print("Pin0 =",Pin0)

# Print the first and last elements
print("First element of New:", New[0])
print("Last element of New:", New[-1])

#Calculate the median of A
median_value = np.median(New)

# Get values greater than the median
values_greater_than_median = New[New > median_value]

# Sort the values
sorted_values = np.sort(values_greater_than_median)

# Print the sorted values in a list
print("Sorted values greater than the median:", list(sorted_values))




first_element = New[0]
last_element = New[-1]
max_first_last = max(first_element, last_element)

Pin0 = [x for x in New if x > max_first_last]

if Pin0:
    print("First element greater than the maximum of the first and last elements:", min(Pin0))
    
    # Use heapq to find the first five minimum elements efficiently
    min_range = heapq.nsmallest(1012, Pin0)
    
    # Create an empty list to store min_val values
    min_values_list = []
    
    for i, min_val in enumerate(min_range, 1):
        print(f"{i} Minimum element in Pin0:", min_val)
        min_values_list.append(min_val)
else:
    print("There are no elements greater than the maximum of the first and last elements in the array")

# Write min_values_list to a file
if min_values_list:
    with open("output_file.txt", "w") as file:
        for min_val in min_values_list:
            file.write(str(min_val) + "\n")

    print("min_values_list has been written to 'output_file.txt'")
else:
    print("min_values_list is empty, nothing to write to the file.")    
#Now min_values_list contains all the min_val values
# print("All min_val values:", min_values_list)    
print("min range here =",min_range)



min_range=[min_range[0]]
#sys.exit()
# Initialize an empty list to store the lengths
#min_range=

for g in range(0,len(min_range)):
    Pin0=min_range[g]
    
    len_J_list = []
    
    try:

        J, Pin0, Total_Indices = Entry_Pressure1(A) 
        print("Pin0 here =",Pin0)
        print("J here =",J)
        
        with open(rf"{Drive}:\Users\debanikb\OneDrive - Technion\Research_Technion\Python_PNM\Surfactant A-D\{Folder_name}\{File}\J_{const}_{power}_initial_{Surf}_{K}_{Cb0}_{Pin0}_{sigma0}.txt", 'w') as f:
#writer = csv.writer(f)   
#print("J =",len(J[0])) 
                    f.writelines('\n')
# if i < len(J[0]) - 1 and len(J[0][t + 1]) >= len(J[0][t]):
#     f.writelines('\n')
#     f.write(str(J[0]))
                    f.write(str(J[0]))

        inv_rJ2 = [inv_r[0][idx] for idx in J[0]]
        inv_rJ1=np.delete(inv_r, J[0])
#print("inv_rJ1 =",inv_rJ1)

        A1=Change_Adjacency(A)
#print("A1 here =",A1)

        X,Flux,Velprof,indices=pressure_flux_velocity()
        # print("Vel prof = ",Velprof)
        # print("indices =",indices)
        # print("Pressure drop =",X)
        
        
#         with open(rf"{Drive}:\Users\debanikb\OneDrive - Technion\Research_Technion\Python_PNM\Surfactant A-D\{Folder_name}\{File}\X_{const}_{power}_initial_{Surf}_{Pin0}.txt", 'w') as f:
# #writer = csv.writer(f)   
# #print("J =",len(J[0])) 
#                     f.writelines('\n')
# # if i < len(J[0]) - 1 and len(J[0][t + 1]) >= len(J[0][t]):
# #     f.writelines('\n')
# #     f.write(str(J[0]))
#                     f.write(str([X]))
                    
                    
#         with open(rf"{Drive}:\Users\debanikb\OneDrive - Technion\Research_Technion\Python_PNM\Surfactant A-D\{Folder_name}\{File}\Velprof_{const}_{power}_initial_{Surf}_{Pin0}.txt", 'w') as f:
# #writer = csv.writer(f)   
# #print("J =",len(J[0])) 
#                     f.writelines('\n')
# # if i < len(J[0]) - 1 and len(J[0][t + 1]) >= len(J[0][t]):
# #     f.writelines('\n')
# #     f.write(str(J[0]))
#                     f.write(str([Velprof]))      
                    
                    
#         with open(rf"{Drive}:\Users\debanikb\OneDrive - Technion\Research_Technion\Python_PNM\Surfactant A-D\{Folder_name}\{File}\Indices_{const}_{power}_initial_{Surf}_{Pin0}.txt", 'w') as f:
# #writer = csv.writer(f)   
# #print("J =",len(J[0])) 
#                     f.writelines('\n')
# # if i < len(J[0]) - 1 and len(J[0][t + 1]) >= len(J[0][t]):
# #     f.writelines('\n')
# #     f.write(str(J[0]))
#                     f.write(str([indices]))                      





#J=[[2, 3, 6, 7, 9, 10]]
        Cb=bulk_conc(Flux,J)
        Cb[0].append(0)
#print("Cb here =",Cb)
        threshold_Cb = Cb0

        for index, value  in enumerate(Cb[0]):
            if value > threshold_Cb:
                Cb[0][index] = threshold_Cb
#print("Cb here 2 =",Cb)

        Cb_max=Cb_max1(A,J,Cb)
#print("Cb_max =",Cb_max)

#Cb=[[0.001, 0.0, 0.0, 0.0, 0.0, 0.0009623159958169187, 0.0, 0.0, 0.0009623159958169193, 0.0, 0.0, 0.0]]

# X=np.array([1.72603703e+003, 0.00000000e+000, 0.00000000e+000, 1.15636944e+003,
#        5.67081351e+002, 1.00000000e-100])

        MaxPrnew,indices_MaxPrnew,indices=local_pressure(indices,Velprof,X,A)
        MaxPrnew = MaxPrnew[0].tolist()
# print("indices =",indices)
# print("Velprof =",Velprof)

   
#print("A here=",A)
        result = [list(np.where(row == 1)[0]) for row in A]
#print("Original result:", result)


# Modify the result based on J
        result_mod = [result[J[0][i]] for i in range(len(J[0]))]
#print("Modified result:", result_mod)
#sys.exit()

        all_max_vel = []
        indices=indices[0]
        for i, result_0_second_elements in enumerate(result_mod):
    
            locations = np.where(np.isin(indices[0][:, 1], result_0_second_elements))
    #print("locations =", locations)

            output = list(zip(indices[0][locations][:, 0], indices[0][locations][:, 1]))
    #print(f"Locations in indices for each element in result[{i}]:", output)
    
    # Identify the locations of output in indices
            output_locations_in_indices = [list(locations[0])]
    #print("Locations of output in indices:", output_locations_in_indices)

    # Convert Velprof values to list
            velprof_values = Velprof[output_locations_in_indices].tolist()
    #print("Velprof values corresponding to output locations in indices:", velprof_values)

            flat_list = [item for sublist in velprof_values for item in sublist]

    # Find the maximum value among all sublists
            max_value = max(flat_list, default=None)  # Set default to None

    # Append the maximum value to the list
            all_max_vel.append(max_value)

#print("All maximum values among all sublists:", all_max_vel)
        all_max_vel = [item if item is not None else 0.0 for item in all_max_vel]  # Replace None with an appropriate default value
#print("all_max_vel =", all_max_vel)

#inv_rJ2=[20208.04178450472, 20384.26891621889, 20141.10609718919, 20323.92940153693]

        #Pe_local = [[(a / b) * (1 / DiffCoeff) for a, b in zip(max_vel, inv_rJ2)] if max_vel else 0.0 for max_vel in all_max_vel]
#print("Pe_local =",Pe_local)

        Pe_local = [[(h / (2*h*b+2)) * (1 / DiffCoeff)*a for a, b in zip(max_vel, inv_rJ2)] if max_vel else 0.0 for max_vel in all_max_vel]
        #print("Pe local =",Pe_local)
        #sys.exit()


        Sh_local = [value[0] ** power if isinstance(value, list) and value and value[0] != 0.0 else 0.0 for value in Pe_local]
        #print("Sh_local =", Sh_local)

        #file_path = rf'{Drive}:\Users\debanikb\OneDrive - Technion\Research_Technion\Python_PNM\Surfactant A-D\{Folder_name}\{File}\Sh_initial_{Pin0}.txt'

# Write Sh_local values to the file in the desired format
        # with open(file_path, 'w') as file:
        #     file.write(f'[{",".join(map(str, Sh_local))}]\n')

#print(f'Sh_local values written to {file_path}')

# MaxPrnew=[241.08095445711552,
#   1726.9903228626436,
#   1232.262749719154,
#   484.3360494454877,
#   1232.262749719154]
#print("Pressure drop, X1 =",[X])

#Sh=12.284
#km=[i*Sh_local*DiffCoeff for i in inv_rJ2]
        #km = [i * Sh * DiffCoeff for i, Sh in zip(inv_rJ2, Sh_local)]
        new_hyd_length=[L*h/(2*L+2*L*h*i) for i in inv_rJ2]
        print(new_hyd_length)
        #sys.exit()
        
        km = [s * DiffCoeff / l for s, l in zip(Sh_local, new_hyd_length)]
        #print(km)
        # print("km =",km)
        # print("inv_rJ2 =",inv_rJ2)
        # size=[1/i for i in inv_rJ2]
        # print("size =",size)
        
        with open(rf"{Drive}:\Users\debanikb\OneDrive - Technion\Research_Technion\Python_PNM\Surfactant A-D\{Folder_name}\{File}\km_{const}_{power}_initial_{Surf}_{K}_{Cb0}_{Pin0}_{sigma0}.txt", 'w') as f:
#writer = csv.writer(f)   
#print("J =",len(J[0])) 
                    f.writelines('\n')
# if i < len(J[0]) - 1 and len(J[0][t + 1]) >= len(J[0][t]):
#     f.writelines('\n')
#     f.write(str(J[0]))
                    f.write(str(km))  
                    
                    
        with open(rf"{Drive}:\Users\debanikb\OneDrive - Technion\Research_Technion\Python_PNM\Surfactant A-D\{Folder_name}\{File}\size_{const}_{power}_initial_{Surf}_{K}_{Cb0}_{Pin0}_{sigma0}.txt", 'w') as f:
#writer = csv.writer(f)   
#print("J =",len(J[0])) 
                    f.writelines('\n')
# if i < len(J[0]) - 1 and len(J[0][t + 1]) >= len(J[0][t]):
#     f.writelines('\n')
#     f.write(str(J[0]))
                    f.write(str(inv_rJ2))    
                    
        with open(rf"{Drive}:\Users\debanikb\OneDrive - Technion\Research_Technion\Python_PNM\Surfactant A-D\{Folder_name}\{File}\Sh_{const}_{power}_initial_{Surf}_{K}_{Cb0}_{Pin0}_{sigma0}.txt", 'w') as f:
#writer = csv.writer(f)   
#print("J =",len(J[0])) 
                    f.writelines('\n')
# if i < len(J[0]) - 1 and len(J[0][t + 1]) >= len(J[0][t]):
#     f.writelines('\n')
#     f.write(str(J[0]))
                    f.write(str(Sh_local))        
                    
                    
        with open(rf"{Drive}:\Users\debanikb\OneDrive - Technion\Research_Technion\Python_PNM\Surfactant A-D\{Folder_name}\{File}\X_{const}_{power}_initial_{Surf}_{K}_{Cb0}_{Pin0}_{sigma0}.txt", 'w') as f:
#writer = csv.writer(f)   
#print("J =",len(J[0])) 
                    f.writelines('\n')
# if i < len(J[0]) - 1 and len(J[0][t + 1]) >= len(J[0][t]):
#     f.writelines('\n')
#     f.write(str(J[0]))
                    f.write(str([X]))
                    
                    
        with open(rf"{Drive}:\Users\debanikb\OneDrive - Technion\Research_Technion\Python_PNM\Surfactant A-D\{Folder_name}\{File}\Velprof_{const}_{power}_initial_{Surf}_{K}_{Cb0}_{Pin0}_{sigma0}.txt", 'w') as f:
#writer = csv.writer(f)   
#print("J =",len(J[0])) 
                    f.writelines('\n')
# if i < len(J[0]) - 1 and len(J[0][t + 1]) >= len(J[0][t]):
#     f.writelines('\n')
#     f.write(str(J[0]))
                    f.write(str([Velprof]))      
                    
                    
        with open(rf"{Drive}:\Users\debanikb\OneDrive - Technion\Research_Technion\Python_PNM\Surfactant A-D\{Folder_name}\{File}\Indices_{const}_{power}_initial_{Surf}_{K}_{Cb0}_{Pin0}_{sigma0}.txt", 'w') as f:
#writer = csv.writer(f)   
#print("J =",len(J[0])) 
                    f.writelines('\n')
# if i < len(J[0]) - 1 and len(J[0][t + 1]) >= len(J[0][t]):
#     f.writelines('\n')
#     f.write(str(J[0]))
                    f.write(str([indices])) 
        
        #sys.exit()

#Cb_max=[0.001, 0.001, 0.0009623159958169193, 0.0009623159958169193, 0.0009623159958169193, 0.0009623159958169193]

#km=[1,2,3,4,5,6]

#km=[3, 5, 6, 2, 1, 2, 5, 1, 4, 5, 5, 8, 3, 8, 3, 2, 6, 1, 10, 3, 3, 9, 7, 4, 3, 10, 6, 1, 8, 6, 4, 2, 7, 5, 5, 9, 7, 4, 2, 4, 10, 9, 6, 2, 6, 7, 6, 2, 1, 5, 7, 2, 10, 1, 2, 7, 7, 4, 5, 2, 6, 7, 1, 10, 8, 4, 1, 8, 2, 3, 5, 7, 6, 2, 8, 8, 3, 4, 3, 3, 8, 3, 9, 4, 10, 7, 4, 3, 6, 5, 3, 4, 4, 9, 4, 2, 2, 3, 2, 5, 6, 3, 10, 4, 9, 7, 3, 3, 3, 10, 1]
    

        # t_start = 0.0
        # t_end = 2e6
        # timesteps = int(0.0005* t_end) + 1
        # t1 = np.linspace(t_start, t_end, timesteps)
        
        
        t_start = 0.0
        t_end = 1e2
        timesteps = int(1e3* t_end) + 1
        t1 = np.linspace(t_start, t_end, timesteps)
        print(timesteps)
#print("t1 =",t1)

        last_J = None
        last_X = None
        last_Velprof = None
        last_indices = None
        row_count=0

######################################################################################################### 
        Indices=[]
        consecutive_nan_count = 0
        first_iteration = True
    
        for t in range(0,len(t1)):
            temp_sigma = []
            temp_Cb_max=[]
            temp_km=[]
            temp_alpha=[]
            temp_Cff=[]
            temp_Ci=[]
            temp_Csf=[]
            Cond_t=[]
            J2=[]
            Pr_cap=[]
    
    
    # print("Indices here =",Indices)
    # print("J here =",J)
    # print("km here =",km)
    #print("Sherwood here 2=",Sh)
            km = [x for i, x in enumerate(km) if i not in Indices]
            #temp_km.append(km)
            
            Pe_local = [[(h / (2*h*b+2)) * (1 / DiffCoeff)*a for a, b in zip(max_vel, inv_rJ2)] if max_vel else 0.0 for max_vel in all_max_vel]
            #print("Pe local =",Pe_local)
            #sys.exit()


            Sh_local = [value[0] ** power if isinstance(value, list) and value and value[0] != 0.0 else 0.0 for value in Pe_local]
            #print("Sh_local =", Sh_local)


            new_hyd_length=[((L*h)/((2*L)+(2*L*h*i))) for i in inv_rJ2]
            print(new_hyd_length)
            #sys.exit()
            
            km = [s * DiffCoeff / l for s, l in zip(Sh_local, new_hyd_length)]
            print("km here =",km)
            
    #         with open(rf"{Drive}:\Users\debanikb\OneDrive - Technion\Research_Technion\Python_PNM\Surfactant A-D\{Folder_name}\{File}\km_{const}_{power}_{t_end}_{timesteps}_{Surf}_{K}_{Cb0}_{Pin0}_{sigma0}.txt", 'a') as f:
    # #writer = csv.writer(f)   
    # #print("J =",len(J[0])) 
    #             f.writelines('\n')
    # # if i < len(J[0]) - 1 and len(J[0][t + 1]) >= len(J[0][t]):
    # #     f.writelines('\n')
    # #     f.write(str(J[0]))
    #             f.write(str(km))
            
            
    #         with open(rf"{Drive}:\Users\debanikb\OneDrive - Technion\Research_Technion\Python_PNM\Surfactant A-D\{Folder_name}\{File}\Cb_{const}_{power}_{t_end}_{timesteps}_{Surf}_{K}_{Cb0}_{Pin0}_{sigma0}.txt", 'a') as f:
    # #writer = csv.writer(f)   
    # #print("J =",len(J[0])) 
    #             f.writelines('\n')
    # # if i < len(J[0]) - 1 and len(J[0][t + 1]) >= len(J[0][t]):
    # #     f.writelines('\n')
    # #     f.write(str(J[0]))
    #             f.write(str(Cb_max))
    
            for i in range(0,len(km)): 
        
                #print("Cb_max here =",Cb_max)
                Ci=Cb_max[i]*(1-np.exp(-km[i]*t1[t]/new_hyd_length[i]))
                #print("Ci here =",Ci)
        
        #print("inv_rJ2 here =",inv_rJ2)
                Cff=(1/(1+(L*inv_rJ2[i])))*Ci
                #print("Cff here =",Cff)
        
                Csf=Ci-Cff
                #print("Csf here =",Csf)
        
                alpha1=alpha0-Csf*mol_wt*100/(rho*1000)
        #print("alpha 1 =",alpha1)
        
                # if alpha1 < threshold_alpha:
                #     alpha1 = threshold_alpha
            
                temp_alpha.append(alpha1)
        
        
                sigma1=sigma0-Rgas*Temp*gamma_max*np.log(1+K*Cff)  #Cb_max[i]
        # print("t1 here =",t1[t])
        #print("sigma 1 =",sigma1)
                # if sigma1 < threshold_sigma:
                #     sigma1 = threshold_sigma
            
            # if alpha1 < threshold_alpha:
            #     alpha1 = threshold_alpha
            
                temp_sigma.append(sigma1)
                #temp_Cb_max.append(Cb_max)
                temp_Ci.append(Ci)
                temp_Cff.append(Cff)
                temp_Csf.append(Csf)
                #temp_km.append(km)
        #temp_alpha.append(alpha1)
            sigma_t.append(temp_sigma)
            alpha_t.append(temp_alpha)
            #km_t.append(temp_km)
            #Cb_max_t.append(temp_Cb_max)
            Ci_t.append(temp_Ci)
            Cff_t.append(temp_Cff)
            Csf_t.append(temp_Csf)
            
    #alpha_t.append(temp_sigma)
    
    # if first_iteration and len(J[0]) != 405:
    #     print("Error: The length of J[0] is not equal to 405. Stopping the code.")
    #     # You can raise an exception or use sys.exit() to stop the code
    #     # For example:
    #     import sys
    #     sys.exit()    
        
    # # Set the flag to False after the first iteration
    # first_iteration = False
    

            # print("sigma t =",sigma_t)
            # print("alpha t =",alpha_t)
            
    #         with open(rf"{Drive}:\Users\debanikb\OneDrive - Technion\Research_Technion\Python_PNM\Surfactant A-D\{Folder_name}\{File}\sigma_{const}_{power}_{t_end}_{timesteps}_{Surf}_{Cb0}_{Pin0}.txt", 'a') as f:
    # #writer = csv.writer(f)   
    # #print("J =",len(J[0])) 
    #             f.writelines('\n')
    # # if i < len(J[0]) - 1 and len(J[0][t + 1]) >= len(J[0][t]):
    # #     f.writelines('\n')
    # #     f.write(str(J[0]))
    #             f.write(str(sigma_t))
                
                
                
    #         with open(rf"{Drive}:\Users\debanikb\OneDrive - Technion\Research_Technion\Python_PNM\Surfactant A-D\{Folder_name}\{File}\alpha_{const}_{power}_{t_end}_{timesteps}_{Surf}_{Cb0}_{Pin0}.txt", 'a') as f:
    # #writer = csv.writer(f)   
    # #print("J =",len(J[0])) 
    #             f.writelines('\n')
    # # if i < len(J[0]) - 1 and len(J[0][t + 1]) >= len(J[0][t]):
    # #     f.writelines('\n')
    # #     f.write(str(J[0]))
    #             f.write(str(alpha_t))
                
                
    #         with open(rf"{Drive}:\Users\debanikb\OneDrive - Technion\Research_Technion\Python_PNM\Surfactant A-D\{Folder_name}\{File}\Cff_{const}_{power}_{t_end}_{timesteps}_{Surf}_{Cb0}_{Pin0}.txt", 'a') as f:
    # #writer = csv.writer(f)   
    # #print("J =",len(J[0])) 
    #             f.writelines('\n')
    # # if i < len(J[0]) - 1 and len(J[0][t + 1]) >= len(J[0][t]):
    # #     f.writelines('\n')
    # #     f.write(str(J[0]))
    #             f.write(str(Cff_t))
                
                
    #         with open(rf"{Drive}:\Users\debanikb\OneDrive - Technion\Research_Technion\Python_PNM\Surfactant A-D\{Folder_name}\{File}\Csf_{const}_{power}_{t_end}_{timesteps}_{Surf}_{Cb0}_{Pin0}.txt", 'a') as f:
    # #writer = csv.writer(f)   
    # #print("J =",len(J[0])) 
    #             f.writelines('\n')
    # # if i < len(J[0]) - 1 and len(J[0][t + 1]) >= len(J[0][t]):
    # #     f.writelines('\n')
    # #     f.write(str(J[0]))
    #             f.write(str(Csf_t))
                
                
                
    #         with open(rf"{Drive}:\Users\debanikb\OneDrive - Technion\Research_Technion\Python_PNM\Surfactant A-D\{Folder_name}\{File}\Ci_{const}_{power}_{t_end}_{timesteps}_{Surf}_{Cb0}_{Pin0}.txt", 'a') as f:
    # #writer = csv.writer(f)   
    # #print("J =",len(J[0])) 
    #             f.writelines('\n')
    # # if i < len(J[0]) - 1 and len(J[0][t + 1]) >= len(J[0][t]):
    # #     f.writelines('\n')
    # #     f.write(str(J[0]))
    #             f.write(str(Ci_t))
                
                
                
            
                
                
    #         with open(rf"{Drive}:\Users\debanikb\OneDrive - Technion\Research_Technion\Python_PNM\Surfactant A-D\{Folder_name}\{File}\km_{const}_{power}_{t_end}_{timesteps}_{Surf}_{Cb0}_{Pin0}.txt", 'a') as f:
    # #writer = csv.writer(f)   
    # #print("J =",len(J[0])) 
    #             f.writelines('\n')
    # # if i < len(J[0]) - 1 and len(J[0][t + 1]) >= len(J[0][t]):
    # #     f.writelines('\n')
    # #     f.write(str(J[0]))
    #             f.write(str(km_t))
    
    
            inv_rJ2 = [x for i, x in enumerate(inv_rJ2) if i not in Indices]
    
            for k in range(0, len(inv_rJ2)):
                Pr_cap.append(-2*sigma_t[t][k] * inv_rJ2[k]*np.cos(alpha_t[t][k] * 180 / np.pi) - 2 *sigma_t[t][k]* (1 / h) * np.cos(alpha_t[t][k] * 180 / np.pi))
    #print("Pr_cap here =",Pr_cap)
    
    
    #print("MaxPrnew here =",MaxPrnew)
    
    #print("J here =",J)
    
            Pr_cap = [Pr_cap[i] for i in range(len(J[0])) if J[0][i] != 0]
            
            
            with open(rf"{Drive}:\Users\debanikb\OneDrive - Technion\Research_Technion\Python_PNM\Surfactant A-D\{Folder_name}\{File}\Pr_cap_{const}_{power}_{t_end}_{timesteps}_{Surf}_{K}_{Cb0}_{Pin0}_{sigma0}.txt", 'a') as f:
        #writer = csv.writer(f)   
        #print("J =",len(J[0])) 
                    f.writelines('\n')
        # if i < len(J[0]) - 1 and len(J[0][t + 1]) >= len(J[0][t]):
        #     f.writelines('\n')
        #     f.write(str(J[0]))
                    f.write(str(Pr_cap))
            
            
    #print("Pr_cap here =",Pr_cap)
            for i in range(0,len(Pr_cap)): 
                Cond1 = MaxPrnew[i] > Pr_cap[i]
                Cond_t.append(Cond1)
    #print("Cond_t=",Cond_t)
    
            Indices=[i for i, v in enumerate(Cond_t) if v]   
    #print("Indices =",Indices)
    
    # with open('Indices.txt', 'a') as f:
    #     #writer = csv.writer(f)   
    #     #print("indices =",indices)
    #     f.writelines('\n')
    #     f.write(str(indices))
    
            J =[x for i, x in enumerate(J[0]) if i not in Indices]
    # J2 = [x if i not in indices else 0 for i, x in enumerate(J[0])]
            J = [J]
            print("len J =",len(J[0]))
            len_J_list.append(len(J[0]))
            print(len_J_list)
            unique_elements = len(set(len_J_list))

            print("Number of unique elements:", unique_elements)
            
            with open(rf"{Drive}:\Users\debanikb\OneDrive - Technion\Research_Technion\Python_PNM\Surfactant A-D\{Folder_name}\{File}\size_{const}_{power}_{t_end}_{timesteps}_{Surf}_{K}_{Pin0}_{sigma0}.txt", 'a') as f:
                         writer = csv.writer(f)
                         f.write(f'[{",".join(map(str, inv_rJ2))}]\n') 
            
            
            
            # Check if any unique element occurs more than 20 times
            if any(len_J_list.count(element) > 100 for element in set(len_J_list)):
                print("At least one unique element occurs more than 100 times. Breaking out of the loop.")
                break  # Use break to exit the loop
            
            
            
            
            
            # if len(J[0]) == 0:
            #     print("Length of J[0] is zero. Terminating further iterations.")
            #     break
    

########################################################################
            
            if len(set(len_J_list)) > 1:
                with open(rf"{Drive}:\Users\debanikb\OneDrive - Technion\Research_Technion\Python_PNM\Surfactant A-D\{Folder_name}\{File}\len_J_{const}_{power}_{t_end}_{timesteps}_{Surf}_{K}_{Cb0}_{Pin0}_{sigma0}.txt", 'a') as f:
        #writer = csv.writer(f)   
        #print("J =",len(J[0])) 
                    f.writelines('\n')
        # if i < len(J[0]) - 1 and len(J[0][t + 1]) >= len(J[0][t]):
        #     f.writelines('\n')
        #     f.write(str(J[0]))
                    f.write(str(len_J_list))
                    
                    
            if 0 in len_J_list:
                    print("Terminating the code due to a zero element in len_J_list.")
        # You may add additional actions before terminating if needed
                    sys.exit()
            
    
    # Count how many rows are in the file
    # with open(rf"{Drive}:\Users\debanikb\OneDrive - Technion\Research_Technion\Python_PNM\Surfactant A-D\{Folder_name}\{File}\len_J_{const}_{power}_{t_end}_{timesteps}_{Surf}_{Cb0}.txt", 'r') as f:
    #     row_count = sum(1 for line in f)

    # print(f"{row_count} rows were written to the file.")


########################################################################

#     file_path = rf"{Drive}:\Users\debanikb\OneDrive - Technion\Research_Technion\Python_PNM\Surfactant A-D\{Folder_name}\{File}\len_J_{const}_{power}_{t_end}_{timesteps}_{Surf}_{Cb0}.txt"

# # Open the file in append mode
#     with open(file_path, 'a') as f:
#     # Limit the number of rows to be written to 25
#         for _ in range(25):
#             f.writelines('\n')
#             f.write(str(len(J[0])))
        
#         # Increment the row counter
#             row_count += 1

#         # Check if 25 rows have been written
#         if row_count == 25:
#             print("Stopping the code after writing 25 rows.")
#             break

# # Count how many rows are in the file
#     with open(file_path, 'r') as f:
#         row_count_in_file = sum(1 for line in f)

#     print(f"{row_count_in_file} rows were written to the file.")
#     print("Closing the file.")
    
########################################################################            
        
            if len(set(len_J_list)) > 1:  
                if J!= last_J:
                        last_J = J
                        with open(rf"{Drive}:\Users\debanikb\OneDrive - Technion\Research_Technion\Python_PNM\Surfactant A-D\{Folder_name}\{File}\J_{const}_{power}_{t_end}_{timesteps}_{Surf}_{K}_{Pin0}_{sigma0}.txt", 'a') as f:
        #writer = csv.writer(f)   
        #print("J =",len(J[0])) 
                            f.writelines('\n')
        # if i < len(J[0]) - 1 and len(J[0][t + 1]) >= len(J[0][t]):
        #     f.writelines('\n')
        #     f.write(str(J[0]))
                            f.write(str(J[0]))
                            
            # if len(set(len_J_list)) > 1:
            #             with open(rf"{Drive}:\Users\debanikb\OneDrive - Technion\Research_Technion\Python_PNM\Surfactant A-D\{Folder_name}\{File}\Sh_{const}_{power}_{t_end}_{timesteps}_{Surf}_{K}_{Pin0}_{sigma0}.txt", 'a') as f:
            #                 writer = csv.writer(f)
            #                 f.write(f'[{",".join(map(str, Sh_local))}]\n')
                        
            # if len(set(len_J_list)) > 1:
            #             with open(rf"{Drive}:\Users\debanikb\OneDrive - Technion\Research_Technion\Python_PNM\Surfactant A-D\{Folder_name}\{File}\size_{const}_{power}_{t_end}_{timesteps}_{Surf}_{K}_{Pin0}_{sigma0}.txt", 'a') as f:
            #                 writer = csv.writer(f)
            #                 f.write(f'[{",".join(map(str, inv_rJ2))}]\n')  
                        
            # if len(set(len_J_list)) > 1:
            #             with open(rf"{Drive}:\Users\debanikb\OneDrive - Technion\Research_Technion\Python_PNM\Surfactant A-D\{Folder_name}\{File}\Pr_cap_{const}_{power}_{t_end}_{timesteps}_{Surf}_{K}_{Pin0}_{sigma0}.txt", 'a') as f:
            #                 writer = csv.writer(f)
            #                 f.write(f'[{",".join(map(str, Pr_cap))}]\n')                             
    
    
    
            A1=Change_Adjacency(A)
            inv_rJ1=np.delete(inv_r, J[0])
    # inv_rJ1=np.array(inv_rJ1)
    #print("inv rJ1 =",inv_rJ1)
    #inv_rJ1=np.array([19371.82250346,19697.49114878, 19899.73292203, 19912.05009437, 19567.94111435,19525.03386252, 19371.82250346])
            X,Flux,Velprof,indices=pressure_flux_velocity()
            #print("X here =",X)
            # Filter out zero elements
            non_zero_X = X[X != 0]

            # Calculate the mean of non-zero elements
            mean_non_zero_X = np.mean(non_zero_X)

            # Print the result
            #print("Mean of non-zero elements in X:", mean_non_zero_X)
            
            with open(rf"{Drive}:\Users\debanikb\OneDrive - Technion\Research_Technion\Python_PNM\Surfactant A-D\{Folder_name}\{File}\X_mean_{const}_{power}_{t_end}_{timesteps}_{Surf}_{K}_{Cb0}_{Pin0}_{sigma0}.txt", 'a') as f:
        #writer = csv.writer(f)   
        #print("J =",len(J[0])) 
                    f.writelines('\n')
        # if i < len(J[0]) - 1 and len(J[0][t + 1]) >= len(J[0][t]):
        #     f.writelines('\n')
        #     f.write(str(J[0]))
                    f.write(str(mean_non_zero_X))
    
    
    #print("Pressure drop, X2=",[X])
    
            Cb=bulk_conc(Flux,J)
            Cb[0].append(0)
    
            for index, value  in enumerate(Cb[0]):
                    if value > threshold_Cb:
                        Cb[0][index] = threshold_Cb
    #print("Cb here 3 =",Cb)
    #print("Cb here =",Cb)

    # print("A here =",A)
    # print("J here =",J)
            Cb_max=Cb_max1(A,J,Cb)
    #print("Cb_max =",Cb_max)
    
    # X= np.array([1.72603703e+003, 0.00000000e+000, 0.00000000e+000, 1.15636944e+003,
    #        5.67081351e+002, 1.00000000e-100])
            MaxPrnew,indices_MaxPrnew,indices=local_pressure(indices,Velprof,X,A)
            MaxPrnew = MaxPrnew[0].tolist()
    #print("MaxPrnew =",MaxPrnew) 
    
    #print("A here=",A)
            result = [list(np.where(row == 1)[0]) for row in A]
    #print("Original result:", result)


    # Modify the result based on J
            result_mod = [result[J[0][i]] for i in range(len(J[0]))]
    #print("Modified result:", result_mod)


            all_max_vel = []
            indices=indices[0]
            for i, result_0_second_elements in enumerate(result_mod):
            
                locations = np.where(np.isin(indices[0][:, 1], result_0_second_elements))
        #print("locations =", locations)

                output = list(zip(indices[0][locations][:, 0], indices[0][locations][:, 1]))
        #print(f"Locations in indices for each element in result[{i}]:", output)
        
        # Identify the locations of output in indices
                output_locations_in_indices = [list(locations[0])]
        #print("Locations of output in indices:", output_locations_in_indices)

        # Convert Velprof values to list
                velprof_values = Velprof[output_locations_in_indices].tolist()
        #print("Velprof values corresponding to output locations in indices:", velprof_values)

                flat_list = [item for sublist in velprof_values for item in sublist]

        # Find the maximum value among all sublists
                max_value = max(flat_list, default=None)  # Set default to None

        # Append the maximum value to the list
                all_max_vel.append(max_value)

    #print("All maximum values among all sublists:", all_max_vel)
            all_max_vel = [item if item is not None else 0.0 for item in all_max_vel]  # Replace None with an appropriate default value
    #print("all_max_vel =", all_max_vel)

    #inv_rJ2=[20208.04178450472, 20384.26891621889, 20141.10609718919, 20323.92940153693]

            Pe_local = [[(h / (2*h*b+2)) * (1 / DiffCoeff)*a for a, b in zip(max_vel, inv_rJ2)] if max_vel else 0.0 for max_vel in all_max_vel]
            #Pe_local =",Pe_local)

            Sh_local = [value[0] ** power if isinstance(value, list) and value and value[0] != 0.0 else 0.0 for value in Pe_local]
    #print("Sh_local =", Sh_local)
    
    #file_path = 'Sh_local_values2.txt'
    
            # if len(set(len_J_list)) > 1:
            #     if last_X is None or not np.array_equal(X, last_X):
            #             last_X = X
            #             with open(rf"{Drive}:\Users\debanikb\OneDrive - Technion\Research_Technion\Python_PNM\Surfactant A-D\{Folder_name}\{File}\X_{const}_{power}_{t_end}_{timesteps}_{Surf}_{K}_{Pin0}_{sigma0}.txt", 'a') as f: 
            #                 writer = csv.writer(f)

    # Write Sh_local values to the file in the desired format
            # if len(set(len_J_list)) > 1:
            #     with open(rf"{Drive}:\Users\debanikb\OneDrive - Technion\Research_Technion\Python_PNM\Surfactant A-D\{Folder_name}\{File}\Sh_{const}_{power}_{t_end}_{timesteps}_{Surf}_{K}_{Pin0}_{sigma0}.txt", 'a') as f:
            #             writer = csv.writer(f)
            #             f.write(f'[{",".join(map(str, Sh_local))}]\n')
                        
            # if len(set(len_J_list)) > 1:
            #     with open(rf"{Drive}:\Users\debanikb\OneDrive - Technion\Research_Technion\Python_PNM\Surfactant A-D\{Folder_name}\{File}\size_{const}_{power}_{t_end}_{timesteps}_{Surf}_{K}_{Pin0}_{sigma0}.txt", 'a') as f:
            #             writer = csv.writer(f)
            #             f.write(f'[{",".join(map(str, inv_rJ2))}]\n')  
                        
            # if len(set(len_J_list)) > 1:
            #     with open(rf"{Drive}:\Users\debanikb\OneDrive - Technion\Research_Technion\Python_PNM\Surfactant A-D\{Folder_name}\{File}\Pr_cap_{const}_{power}_{t_end}_{timesteps}_{Surf}_{K}_{Pin0}_{sigma0}.txt", 'a') as f:
            #             writer = csv.writer(f)
            #             f.write(f'[{",".join(map(str, Pr_cap))}]\n')                             

    #print(f'Sh_local values written to {file_path}')
    
    #X= [1.72603703e+003,0.00000000e+000,0.00000000e+000,1.15636944e+003,5.67081351e+002,1.00000000e-100]
    
            if len(set(len_J_list)) > 1:
                if last_X is None or not np.array_equal(X, last_X):
                        last_X = X
                        with open(rf"{Drive}:\Users\debanikb\OneDrive - Technion\Research_Technion\Python_PNM\Surfactant A-D\{Folder_name}\{File}\X_{const}_{power}_{t_end}_{timesteps}_{Surf}_{K}_{Pin0}_{sigma0}.txt", 'a') as f: 
                            writer = csv.writer(f)
        # writer.writerow(['Pressure drop','Indices','Velocity profile']) 
        #for x in range(0,Total):
        #print(X)
        #print(sigma)  
        # writer.writerows(zip([[X]],[J],
        #                       [[Velprof]]))  
        
                            f.writelines('\n')
                            f.write(str([X]))
            
            
    # with open(rf"{Drive}:\Users\debanikb\OneDrive - Technion\Research_Technion\Python_PNM\Surfactant A-D\{Folder_name}\{File}\Peclet_{const}_{power}_{t_end}_{timesteps}_{Surf}.txt", 'a') as f: 
    #     writer = csv.writer(f)
    # # writer.writerow(['Pressure drop','Indices','Velocity profile']) 
    # #for x in range(0,Total):
    # #print(X)
    # #print(sigma)  
    # # writer.writerows(zip([[X]],[J],
    # #                       [[Velprof]]))  
    
    #     f.writelines('\n')
    #     f.write(str([Peclet]))
        
        
    # with open(rf"{Drive}:\Users\debanikb\OneDrive - Technion\Research_Technion\Python_PNM\Surfactant A-D\{Folder_name}\{File}\Sherwood_{const}_{power}_{t_end}_{timesteps}_{Surf}.txt", 'a') as f: 
    #     writer = csv.writer(f)
    # # writer.writerow(['Pressure drop','Indices','Velocity profile']) 
    # #for x in range(0,Total):
    # #print(X)
    # #print(sigma)  
    # # writer.writerows(zip([[X]],[J],
    # #                       [[Velprof]]))  
    
    #     f.writelines('\n')
    #     f.write(str([Sh]))
    
    # with open(rf"C:\Users\User\OneDrive - Technion\Research_Technion\Python_PNM\Surfactant A-D\12 nodes_New\{File}\X.txt", 'a') as f: 
    #     writer = csv.writer(f)
    #     # writer.writerow(['Pressure drop','Indices','Velocity profile']) 
    #     #for x in range(0,Total):
    #     #print(X)
    #     #print(sigma)  
    #     # writer.writerows(zip([[X]],[J],
    #     #                       [[Velprof]]))  
        
    #     f.writelines('\n')
    #     f.write(str([X]))
        
        
        #writer.writerow([X])
            if len(set(len_J_list)) > 1:
                if last_Velprof is None or not np.array_equal(Velprof, last_Velprof):
                        last_Velprof = Velprof
                        with open(rf"{Drive}:\Users\debanikb\OneDrive - Technion\Research_Technion\Python_PNM\Surfactant A-D\{Folder_name}\{File}\Velprof_{const}_{power}_{t_end}_{timesteps}_{Surf}_{K}_{Pin0}_{sigma0}.txt", 'a') as f: 
                            writer = csv.writer(f)
        # writer.writerow(['Pressure drop','Indices','Velocity profile']) 
        #for x in range(0,Total):
        #print(X)
        #print(sigma)  
        # writer.writerows(zip([[X]],[J],
        #                       [[Velprof]]))  
        
                            f.writelines('\n')
                            f.write(str([Velprof]))
        
            
            if len(set(len_J_list)) > 1:
                if last_indices is None or not np.array_equal(indices, last_indices):
                        last_indices = indices 
                        with open(rf"{Drive}:\Users\debanikb\OneDrive - Technion\Research_Technion\Python_PNM\Surfactant A-D\{Folder_name}\{File}\Indices_{const}_{power}_{t_end}_{timesteps}_{Surf}_{K}_{Pin0}_{sigma0}.txt", 'a') as f: 
                            writer = csv.writer(f)
        # writer.writerow(['Pressure drop','Indices','Velocity profile']) 
        #for x in range(0,Total):
        #print(X)
        #print(sigma)  
        # writer.writerows(zip([[X]],[J],
        #                       [[Velprof]]))  
        
                            f.writelines('\n')
                            f.write(str([indices]))
                            
    #         with open(rf"{Drive}:\Users\debanikb\OneDrive - Technion\Research_Technion\Python_PNM\Surfactant A-D\{Folder_name}\{File}\sigma_{const}_{power}_{t_end}_{timesteps}_{Surf}_{K}_{Cb0}_{Pin0}_{sigma0}.txt", 'a') as f:
    # #writer = csv.writer(f)   
    # #print("J =",len(J[0])) 
    #             f.writelines('\n')
    # # if i < len(J[0]) - 1 and len(J[0][t + 1]) >= len(J[0][t]):
    # #     f.writelines('\n')
    # #     f.write(str(J[0]))
    #             f.write(str(sigma_t))
                
                
                
    #         with open(rf"{Drive}:\Users\debanikb\OneDrive - Technion\Research_Technion\Python_PNM\Surfactant A-D\{Folder_name}\{File}\alpha_{const}_{power}_{t_end}_{timesteps}_{Surf}_{K}_{Cb0}_{Pin0}_{sigma0}.txt", 'a') as f:
    # #writer = csv.writer(f)   
    # #print("J =",len(J[0])) 
    #             f.writelines('\n')
    # # if i < len(J[0]) - 1 and len(J[0][t + 1]) >= len(J[0][t]):
    # #     f.writelines('\n')
    # #     f.write(str(J[0]))
    #             f.write(str(alpha_t))
                
                
    #         with open(rf"{Drive}:\Users\debanikb\OneDrive - Technion\Research_Technion\Python_PNM\Surfactant A-D\{Folder1}\{Folder_name}\{File}\Cff_{const}_{power}_{t_end}_{timesteps}_{Surf}_{K}_{Cb0}_{Pin0}_{sigma0}.txt", 'a') as f:
    # #writer = csv.writer(f)   
    # #print("J =",len(J[0])) 
    #             f.writelines('\n')
    # # if i < len(J[0]) - 1 and len(J[0][t + 1]) >= len(J[0][t]):
    # #     f.writelines('\n')
    # #     f.write(str(J[0]))
    #             f.write(str(Cff_t))
                
                
    #         with open(rf"{Drive}:\Users\debanikb\OneDrive - Technion\Research_Technion\Python_PNM\Surfactant A-D\{Folder1}\{Folder_name}\{File}\Csf_{const}_{power}_{t_end}_{timesteps}_{Surf}_{K}_{Cb0}_{Pin0}_{sigma0}.txt", 'a') as f:
    # #writer = csv.writer(f)   
    # #print("J =",len(J[0])) 
    #             f.writelines('\n')
    # # if i < len(J[0]) - 1 and len(J[0][t + 1]) >= len(J[0][t]):
    # #     f.writelines('\n')
    # #     f.write(str(J[0]))
    #             f.write(str(Csf_t))
                
                
                
    #         with open(rf"{Drive}:\Users\debanikb\OneDrive - Technion\Research_Technion\Python_PNM\Surfactant A-D\{Folder1}\{Folder_name}\{File}\Ci_{const}_{power}_{t_end}_{timesteps}_{Surf}_{K}_{Cb0}_{Pin0}_{sigma0}.txt", 'a') as f:
    # #writer = csv.writer(f)   
    # #print("J =",len(J[0])) 
    #             f.writelines('\n')
    # # if i < len(J[0]) - 1 and len(J[0][t + 1]) >= len(J[0][t]):
    # #     f.writelines('\n')
    # #     f.write(str(J[0]))
    #             f.write(str(Ci_t))


        # arJ.append(J)
        # J=list(arJ)
        
            
            # if unique_elements == 1000:
            #     print("Terminating the code as there are many unique elements.")
            #     break
        
            
        
        

        U=time.time() - start_time
        if len(set(len_J_list)) > 1: 
            with open(rf"{Drive}:\Users\debanikb\OneDrive - Technion\Research_Technion\Python_PNM\Surfactant A-D\{Folder_name}\{File}\Time_{const}_{power}_{t_end}_{timesteps}_{Surf}_{K}_{Pin0}_{sigma0}.csv", 'a') as f:
                    writer = csv.writer(f)
    #print("Time (seconds) =",U)
                    f.writelines('Time (minutes)'+ '\n')
                    f.write(str(U/60))
    

        
    
        filepath = rf"{Drive}:\Users\debanikb\OneDrive - Technion\Research_Technion\Python_PNM\Surfactant A-D\{Folder_name}\{File}\len_J_{const}_{power}_{t_end}_{timesteps}_{Surf}_{K}_{Cb0}_{Pin0}_{sigma0}.txt"

    # Read and print the contents of the file
        try:
            with open(filepath, 'r') as file:
                file_contents = file.read()
                print("File Contents:")
                print(file_contents)

            # Check if all elements in the file are the same
                elements = file_contents.split()  # Assuming elements are separated by spaces

                if all(element == elements[0] for element in elements):
                    print("All elements in the file are the same.")
                else:
                    print("Not all elements in the file are the same.")
                    sys.exit()
                    
        except FileNotFoundError:
            print(f"Error: File not found at {filepath}")
        except Exception as e:
            print(f"An error occurred: {e}")
            
    except Exception as e: 
                print(f"An error occurred in iteration {g}: {e}")
                
                
    
                

        
    
# Print the list of lengths after all iterations
print("Lengths of J[0] for each iteration:", len_J_list)
        
    
    
       
    
        
