from joblib import Parallel, delayed
import time, math

def my_fun(i):
    """ We define a simple function here.
    """
    time.sleep(1)
    return math.sqrt(i**2)
	
num = 10
start = time.time()
# n_jobs is the number of parallel jobs
Parallel(n_jobs=-1)(delayed(my_fun)(i) for i in range(num))
end = time.time()
print('{:.4f} s'.format(end-start))

in_channels: int, out_channels: int,
                 aggr: str = 'softmax', t: float = 1.0, learn_t: bool = False,
                 p: float = 1.0, learn_p: bool = False, msg_norm: bool = False,
                 learn_msg_scale: bool = False, norm: str = 'batch',
                 num_layers: int = 2, eps: float = 1e-7, **kwargs)