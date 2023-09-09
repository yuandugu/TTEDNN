from mpi4py import MPI
import math
import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import levy_stable
import random
import h5py

#####################  parameters  ####################

N = 14                           # number of node
alpha = 0.1                      # damping
theta = math.pi                  # range of theta_0
omega = 20                       # range of omega_0
step = 0.01                      # time step
max_t = 120                      # maximum time
t = np.arange(0, max_t, step)    # time stream
data_number = 1000               # samping number
interval = False
if interval:
    cut_out_num = 51             # collect data number, 101 for 14, 51 for 39
else:
    cut_out_num = 101


def dmove(t, y, sets):
    """
    swing equations definition
    """
    X = np.zeros((N * 2))
    for i in range(N):
        X[i] = y[i + N]
        a = 0
        for j in range(N):
            a += sets[i + 1, j] * math.sin(y[j] - y[i])
        X[i + N] = -alpha * y[i + N] + sets[0, i] + a
    return X


def load_para():

    f = h5py.File('parameter/parameter%s.h5' % (N), 'r')
    PY = f['PY'][()]
    initial = f['initial'][()]
    f.close()
    print('Electrical parameters loaded.')
    return PY, initial


def generate_uniform_init_array(Initial, init_num, node_num):
    """
    generate uniform-distributed samples
    """
    np.random.seed(node_num*570)
    init_array = np.random.rand(2, init_num)
    init_array -= 0.5*np.ones((2, init_num))
    init_array[0, :] *= 2 * theta
    init_array[0, :] += Initial[node_num - 1] * np.ones((init_num))
    init_array[1, :] *= 2 * omega
    return init_array


def generate_stable_init_array(Initial, init_num, node_num):
    """
    generate levy_stable-distributed samples
    """
    alpha, beta, loc, scale = 1.97, 0, 0, 2.42*2
    np.random.seed(node_num*570)
    init_array = np.zeros((2, init_num))
    init_array[0, :] = np.random.rand(init_num)
    init_array[0, :] -= 0.5*np.ones((init_num))
    init_array[0, :] *= 2 * theta
    init_array[0, :] += Initial[node_num - 1] * np.ones((init_num))
    init_array[1, :] = np.clip(levy_stable.rvs(alpha, beta, loc, scale, size=init_num), -omega, omega)
    return init_array


def solve(i, PY, initial):
    """
    parallel function
    """
    names = locals()
    a = np.hstack((initial, np.zeros(N)))  # IEEE-14 equilibrium
    names['init_'+str(i)] = generate_uniform_init_array(
        Initial=a, init_num=data_number, node_num=i + 1
    )
    S = []
    data_theta = np.zeros((data_number, cut_out_num * N))
    data_omega = np.zeros((data_number, cut_out_num * N))
    for j in range(data_number):
        init = a
        init[i] = names['init_'+str(i)][0, j]
        init[i+N] = names['init_'+str(i)][1, j]
        names['result' + str(i) + str(j)] = solve_ivp(
            fun=lambda t, y: dmove(t, y, PY),
            t_span=(0.0, max_t),  y0=init, method='RK45', t_eval=t
        )
        for num in range(N):
            if interval:
                data_theta[j, num*cut_out_num:(num*cut_out_num+cut_out_num)] = names['result' + str(i) + str(j)].y[num, 0:4*cut_out_num-3:4]
                data_omega[j, num*cut_out_num:(num*cut_out_num+cut_out_num)] = names['result' + str(i) + str(j)].y[num+N, 0:4*cut_out_num-3:4]
            else:
                data_theta[j, num*cut_out_num:(num*cut_out_num+cut_out_num)] = names['result' + str(i) + str(j)].y[num, 0:cut_out_num]
                data_omega[j, num*cut_out_num:(num*cut_out_num+cut_out_num)] = names['result' + str(i) + str(j)].y[num+N, 0:cut_out_num]
        if(np.amax(abs(names['result' + str(i) + str(j)].y[N:, -1])) <= 0.2):
            S.append(0)
        else:
            S.append(1)

        del names['result' + str(i) + str(j)], init
        print('(%s,%s) done' % (i+1, j+1))
    if interval:
        f = h5py.File('single/%s.h5' % (i+1), 'w')
    else:
        f = h5py.File('single/%s.h5' % (i+1), 'w')
    f.create_dataset('data_theta', data=data_theta)
    f.create_dataset('data_omega', data=data_omega)
    f.create_dataset('Y', data=np.array(S))
    f.close()


def parallel(PY, initial):
    """
    parallel calculation
    """
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    numjobs = N

    job_content = []
    for i_cur in range(N):
        job_content.append(i_cur)

    if rank == 0:
        job_all_idx = list(range(numjobs))
        random.shuffle(job_all_idx)
    else:
        job_all_idx = None

    job_all_idx = comm.bcast(job_all_idx, root=0)

    njob_per_worker, res = divmod(numjobs, size)
    if rank < res:
        this_worker_job = [job_all_idx[x] for x in range(rank*(njob_per_worker + 1), (rank + 1) * (njob_per_worker + 1))]
    elif rank >= res:
        this_worker_job = [job_all_idx[x] for x in range(rank*njob_per_worker + res, (rank + 1) * njob_per_worker + res)]

    work_content = [job_content[x] for x in this_worker_job]

    for a_piece_of_work in work_content:
        print('core number:%s' % (rank))
        solve(i=a_piece_of_work, PY=PY, initial=initial)


if __name__ == "__main__":

    PY, initial = load_para()
    parallel(PY=PY, initial=initial)
