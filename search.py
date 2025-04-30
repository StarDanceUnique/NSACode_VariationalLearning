# This script searches for the optimal parameters for a given QECC code configuration. 
# Users must specify the noise model and error set. Additionally, they need to adjust 
# the seed, gamma, and function tolerance (fun_tol) for both L1 and L2 loss functions.
import numpy as np
from scipy.optimize import minimize
from numpy import inf
import pickle
import time
import os

from utils import parse_qecc_str, loss_function_wrapper, make_all_to_all_connection, Debbie_amplitude_damping_noise


class Trigger(Exception):
    pass


class ObjectiveFunctionWrapper:

    def __init__(self, fun, fun_tol=None, pfreq=100):
        self.fun = fun
        self.best_x = None
        self.best_f = inf
        self.fun_tol = fun_tol or -inf
        self.number_of_f_evals = 0
        self.pfreq = pfreq
        self.start_time = time.time()  # Start time
        self.time_history = []  # List to store time history
        print(f'fun_tol = {self.fun_tol}')  

    def __call__(self, x):
        _f = self.fun(x)
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        self.time_history.append(elapsed_time)
        
        if self.number_of_f_evals % self.pfreq == 0:
            print(f'[step={self.number_of_f_evals}][time={elapsed_time:.3f} seconds] current value = {_f}')
            folder = 'parameters'
            file_name = f'current_parameters_n{num_qubit}_K{num_logical_dim}_t{t}_L{num_layer}.pkl'
            file_path = os.path.join(folder, file_name)
            with open(file_path, 'wb') as file:
                pickle.dump(self.best_x, file)

        self.number_of_f_evals += 1

        if _f < self.best_f:
            self.best_x, self.best_f = x, _f

        return _f

    def stop(self, *args):
        if self.best_f < self.fun_tol:
            raise Trigger
        # if self.number_of_f_evals > 100000:
            # print('Exceeded max number of iterations', self.number_of_f_evals)
            # raise Trigger


""" initialize the qecc parameters, note that K is the logical qubit dimension """
qecc_str = '((4,2,1))[3]'  #((n,K,t))[L]

tmp0 = parse_qecc_str(qecc_str)
print(tmp0)
num_qubit = tmp0['num_qubit']
num_logical_dim = tmp0['num_logical_dim']
t = tmp0['weight']
num_layer = tmp0['num_layer']


""" noise strength """
gamma = 10**-1.7


""" generate connection list """
connection_list = make_all_to_all_connection(num_qubit)

""" generate amplitude damping errors """
error_list = Debbie_amplitude_damping_noise(num_qubit, t, gamma)
print('total number of errors:', len(error_list))

""" initialize loss funcions and parameters """
hf_loss_L2, num_parameter = loss_function_wrapper(num_qubit,
        num_logical_dim, error_list, num_layer, connection_list, loss_type='L2')
hf_loss_L1, num_parameter = loss_function_wrapper(num_qubit,
        num_logical_dim, error_list, num_layer, connection_list, loss_type='L1')


""" run vqc """
seed = 340721
np_rng = np.random.default_rng(seed)


fun_tol = gamma ** 2.5
theta0 = np_rng.uniform(0, 2*np.pi, num_parameter)
f_wrapped_L2 = ObjectiveFunctionWrapper(fun=hf_loss_L2, fun_tol=fun_tol)

try:
        minimize(
                f_wrapped_L2,
                theta0,  
                method="BFGS",
                callback=f_wrapped_L2.stop
        )
except Trigger:
        print(f"Found f value below tolerance of {fun_tol}\
                in {f_wrapped_L2.number_of_f_evals} f-evals:\
                \nf(x) = {f_wrapped_L2.best_f}\
                \n")
except Exception as e:
        raise e

fun_tol = gamma ** 4 # need to change this sometimes
theta_0 = f_wrapped_L2.best_x
f_wrapped_L1 = ObjectiveFunctionWrapper(fun=hf_loss_L1, fun_tol=fun_tol)

# print('PASSED THE FIRST ROUND OF OPTIMIZATION \n')

try:
        minimize(
                f_wrapped_L1,
                theta_0,
                #method = "L-BFGS-B",  
                method="BFGS",
                #method="Powell",
                #method="SLSQP",
                callback=f_wrapped_L1.stop
        )
except Trigger:
        print(f"Found f value below tolerance of {fun_tol}\
                in {f_wrapped_L1.number_of_f_evals} f-evals:\
                \nf(x) = {f_wrapped_L1.best_f}")
except Exception as e:
        raise e


print(f'L1:{f_wrapped_L1.best_f}')

file_name = f'best_theta_parameters_n{num_qubit}_K{num_logical_dim}_t{t}_L{num_layer}_step{f_wrapped_L1.number_of_f_evals}.pkl'
folder = 'parameters'
file_path = os.path.join(folder, file_name)
with open(file_path, 'wb') as file:
    pickle.dump(f_wrapped_L1.best_x, file)

print(f"Best parameters saved to {file_name}")