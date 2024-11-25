from scipy.optimize import minimize, LinearConstraint
import numpy as np
import logging
import time

class resource_allocation:
    @classmethod
    def resource_allocate_opt(cls, Bandwidth, FLOPs, sum_dataset, parameters, h, FLOPS, p_min=0.01, p_max=46):
        subcarrier_num = h.__len__()
        user_per_sbc = []
        for i in range(subcarrier_num):
            user_per_sbc.append(np.shape(h[i])[0])
        user_num = sum(user_per_sbc)

        def cost(power_sbcbw):
            subcarrier_bandwidth = power_sbcbw[user_num:].reshape(subcarrier_num)
            power_flat = power_sbcbw[:user_num]
            power, A, y, uplink_time,pointer=[], [], [], [], 0
            for i in range(subcarrier_num):
                power.append(power_flat[pointer:pointer+user_per_sbc[i]])
                pointer+=user_per_sbc[i]
                A.append(h[i]*power[-1])
                y.append(np.zeros(user_per_sbc[i], dtype=float))
                uplink_time.append(np.zeros(user_per_sbc[i], dtype=float))

            for i in range(subcarrier_num):
                for j in range(user_per_sbc[i]):
                    y[i][j] = A[i][j] + y[i][j-1]
            for i in range(subcarrier_num):
                for j in range(user_per_sbc[i]):
                    y[i][j] = subcarrier_bandwidth[i] * subcarrier_num / Bandwidth + y[i][j]
            for i in range(subcarrier_num):
                for j in range(user_per_sbc[i]):
                    uplink_time[i][j] = parameters / (subcarrier_bandwidth[i] * (np.log2(y[i][j] / (y[i][j] - A[i][j]))))
            cvx_U_uplink = 0
            for i in range(subcarrier_num):
                for j in range(user_per_sbc[i]):
                    cvx_U_uplink += FLOPS[i][j] * uplink_time[i][j]

            cvx_U_uplink /= (FLOPs * sum_dataset)
            # cvx_U_uplink = sum(sum(FLOPS * uplink_time)) / (FLOPs * sum_dataset)
            return cvx_U_uplink

        def bounds():
            bnds = ((p_min, p_max),)*user_num + ((0,Bandwidth),)*subcarrier_num
            return bnds

        def con():
            def con_1(x):
                su=Bandwidth
                for i in range(subcarrier_num):
                    su-=x[user_num+i]
                # print(su)
                return su
            cons = {'type': 'eq', 'fun': con_1}
            return cons

        cons = con()
        bnds = bounds()
        # 设置初始猜测值
        # x0 = np.array([1]*user_num + [(200)/2]*2 +[Bandwidth-200])
        x0 = np.array([1]*user_num + [Bandwidth/subcarrier_num]*subcarrier_num)
        res = minimize(cost, x0, method='SLSQP', constraints=cons, bounds=bnds, options={'maxiter':100000000, 'ftol': 1e-50, 'disp':True})
        # res = minimize(cost, x0, method='COBYLA', constraints=cons)
        # print(res.fun)
        # print(res.success)
        # print('upperbound')
        print(res.x)
        return res
    @classmethod
    def resource_allocate_ineq(cls, Bandwidth, FLOPs, sum_dataset, parameters, h, FLOPS, p_min=0.01, p_max=40):
        subcarrier_num = h.__len__()
        user_per_sbc = []
        for i in range(subcarrier_num):
            user_per_sbc.append(np.shape(h[i])[0])

        def cost(Ap_sbcbw):
            A_p = Ap_sbcbw[:subcarrier_num]
            subcarrier_bandwidth = Ap_sbcbw[subcarrier_num:]
            pmax = p_max
            uplink_ineq_time = np.zeros(subcarrier_num)
            ineq_U_uplink = 0
            for i in range(subcarrier_num):
                F_sqrt_sum = 0
                for j in range(user_per_sbc[i]-1):
                    F_sqrt_sum = F_sqrt_sum + FLOPS[i][j] ** 0.5
                uplink_ineq_time[i] = FLOPS[i][user_per_sbc[i]-1] / np.log2(
                    (A_p[i] + pmax * h[i][user_per_sbc[i]-1]) / A_p[i]) + F_sqrt_sum ** 2 / (
                                                  np.log2(A_p[i]) - np.log2(
                                              subcarrier_bandwidth[i] * subcarrier_num / Bandwidth))
            for i in range(subcarrier_num):
                ineq_U_uplink = ineq_U_uplink + parameters * uplink_ineq_time[i] / subcarrier_bandwidth[i]
            ineq_U_uplink = ineq_U_uplink / (FLOPs * sum_dataset)
            # print(ineq_U_uplink)
            return ineq_U_uplink

        def bounds():
            bnds = None
            for i in range(subcarrier_num):
                bnds = ((p_min, p_max * h[i][user_per_sbc[i]-1]), ) if bnds is None else \
                    bnds + ((p_min, p_max * h[i][user_per_sbc[i]-1]), )
            for i in range(subcarrier_num):
                bnds = bnds + ((0, Bandwidth),)
            # print(bnds)
            return bnds

        def con():
            def con_1(x):
                su=Bandwidth
                for i in range(subcarrier_num):
                    su-=x[subcarrier_num+i]
                return su
            def con_2(x):
                return min(x[:subcarrier_num] - 1 - x[subcarrier_num:] * subcarrier_num / Bandwidth)
            cons = [{'type': 'eq', 'fun': con_1}, {'type': 'ineq', 'fun': con_2}, ]
            return cons

        x0 = np.array([20]*subcarrier_num + [Bandwidth/subcarrier_num]*subcarrier_num)

        cons = con()
        bnds = bounds()

        res = minimize(cost, x0, method='SLSQP', constraints=cons, bounds=bnds, options={'maxiter':100000000, 'ftol': 1e-50, 'disp':True})
        print(res.x)
        return res
    @classmethod
    def power_allocate_opt(cls, Bandwidth, FLOPs, sum_dataset, parameters, h, FLOPS, p_min=0.01, p_max=46):
        subcarrier_num = h.__len__()
        user_per_sbc = []
        for i in range(subcarrier_num):
            user_per_sbc.append(np.shape(h[i])[0])
        user_num = sum(user_per_sbc)

        def cost(power_sbcbw):
            subcarrier_bandwidth = np.array([Bandwidth / subcarrier_num] * subcarrier_num)
            power_flat = power_sbcbw[:user_num]
            power, A, y, uplink_time, pointer = [], [], [], [], 0
            for i in range(subcarrier_num):
                power.append(power_flat[pointer:pointer + user_per_sbc[i]])
                pointer += user_per_sbc[i]
                A.append(h[i] * power[-1])
                y.append(np.zeros(user_per_sbc[i], dtype=float))
                uplink_time.append(np.zeros(user_per_sbc[i], dtype=float))
            for i in range(subcarrier_num):
                for j in range(user_per_sbc[i]):
                    y[i][j] = A[i][j] + y[i][j - 1]
            for i in range(subcarrier_num):
                for j in range(user_per_sbc[i]):
                    y[i][j] = subcarrier_bandwidth[i] * subcarrier_num / Bandwidth + y[i][j]
            for i in range(subcarrier_num):
                for j in range(user_per_sbc[i]):
                    uplink_time[i][j] = parameters / (
                            subcarrier_bandwidth[i] * (np.log2(y[i][j] / (y[i][j] - A[i][j]))))
            cvx_U_uplink = 0
            for i in range(subcarrier_num):
                for j in range(user_per_sbc[i]):
                    cvx_U_uplink += FLOPS[i][j] * uplink_time[i][j]
            cvx_U_uplink /= (FLOPs * sum_dataset)
            return cvx_U_uplink

        def bounds():
            bnds = ((p_min, p_max),) * user_num
            return bnds
        bnds = bounds()
        # 设置初始猜测值
        x0 = np.array([1] * user_num)
        res = minimize(cost, x0, method='SLSQP', bounds=bnds,
                       options={'maxiter': 100000000, 'ftol': 1e-50, 'disp': True})
        # print(res.x)
        # print(Bandwidth / subcarrier_num)
        return res

def Uniform_distribution(Bandwidth, FLOPs, sum_dataset, parameters, h, FLOPS,time_of_a_round,dataset, p_max=40):
    subcarrier_num = h.__len__()
    user_per_sbc = []
    power, y, A, uplink_time,iteration = [], [], [], [],[]
    for i in range(subcarrier_num):
        user_per_sbc.append(np.shape(h[i])[0])
        power.append(np.zeros(user_per_sbc[i]))
        y.append(np.zeros(user_per_sbc[i]))
        uplink_time.append(np.zeros(user_per_sbc[i]))
        iteration.append(np.zeros(user_per_sbc[i]))
    user_num = sum(user_per_sbc)
    subcarrier_bandwidth = np.array([Bandwidth / subcarrier_num] * subcarrier_num)

    for i in range(subcarrier_num):
        for j in range(user_per_sbc[i]):
            power[i][j]=p_max*(1+j)/user_per_sbc[i]
        A.append(h[i] * power[i])
    for i in range(subcarrier_num):
        for j in range(user_per_sbc[i]):
            y[i][j] = A[i][j] + y[i][j - 1]
    for i in range(subcarrier_num):
        for j in range(user_per_sbc[i]):
            y[i][j] = subcarrier_bandwidth[i] * subcarrier_num / Bandwidth + y[i][j]
    for i in range(subcarrier_num):
        for j in range(user_per_sbc[i]):
            uplink_time[i][j] = parameters / (subcarrier_bandwidth[i] * (np.log2(y[i][j] / (y[i][j] - A[i][j]))))
    cvx_U_uplink = 0
    for i in range(subcarrier_num):
        for j in range(user_per_sbc[i]):
            cvx_U_uplink += FLOPS[i][j] * uplink_time[i][j]
            iteration[i][j] = (time_of_a_round - uplink_time[i][j]) * FLOPS[i][j] / (FLOPs * dataset[i][j])
            if iteration[i][j]<1:
                iteration[i][j]=1
    cvx_U_uplink /= (FLOPs * sum_dataset)
    # print(cvx_U_uplink)
    # print('uniform iteration')
    iteration_flat = None
    for i in range(subcarrier_num):
        iteration_flat = iteration[i] if iteration_flat is None else np.concatenate((iteration_flat, iteration[i]),
                                                                                    axis=0)
    # print(iteration_flat)
    return cvx_U_uplink,iteration_flat
def Power_max(Bandwidth, FLOPs, sum_dataset, parameters, h, FLOPS, time_of_a_round,dataset,p_max=40):
    subcarrier_num = h.__len__()
    user_per_sbc = []
    power,y,A,uplink_time,iteration=[],[],[],[],[]
    for i in range(subcarrier_num):
        user_per_sbc.append(np.shape(h[i])[0])
        power.append(np.zeros(user_per_sbc[i]))
        y.append(np.zeros(user_per_sbc[i]))
        iteration.append(np.zeros(user_per_sbc[i]))
        uplink_time.append(np.zeros(user_per_sbc[i]))
    subcarrier_bandwidth =  np.array([Bandwidth/subcarrier_num]*subcarrier_num)
    user_num = sum(user_per_sbc)
    for i in range(subcarrier_num):
        for j in range(user_per_sbc[i]):
            power[i][j]=p_max
        A.append(h[i] * power[i])
    for i in range(subcarrier_num):
        for j in range(user_per_sbc[i]):
            y[i][j] = A[i][j] + y[i][j - 1]
    for i in range(subcarrier_num):
        for j in range(user_per_sbc[i]):
            y[i][j] = subcarrier_bandwidth[i] * subcarrier_num / Bandwidth + y[i][j]
    for i in range(subcarrier_num):
        for j in range(user_per_sbc[i]):
            uplink_time[i][j] = parameters / (subcarrier_bandwidth[i] * (np.log2(y[i][j] / (y[i][j] - A[i][j]))))
    cvx_U_uplink = 0
    for i in range(subcarrier_num):
        for j in range(user_per_sbc[i]):
            cvx_U_uplink += FLOPS[i][j] * uplink_time[i][j]
            iteration[i][j] = (time_of_a_round - uplink_time[i][j]) * FLOPS[i][j] / (FLOPs * dataset[i][j])
            if iteration[i][j]<1:
                iteration[i][j]=1
    cvx_U_uplink /= (FLOPs * sum_dataset)
    # print(cvx_U_uplink)
    # print('Power_max')
    iteration_flat = None
    for i in range(subcarrier_num):
        iteration_flat = iteration[i] if iteration_flat is None else np.concatenate((iteration_flat, iteration[i]),
                                                                                    axis=0)
    # print(iteration_flat)
    return cvx_U_uplink,iteration_flat
def TDMA(Bandwidth, FLOPs, sum_dataset, parameters, h, FLOPS,time_of_a_round,dataset, p_max=40):
    subcarrier_num = h.__len__()
    user_per_sbc,uplink_time,iteration = [],[],[]
    for i in range(subcarrier_num):
        user_per_sbc.append(np.shape(h[i])[0])
        uplink_time.append(np.zeros(user_per_sbc[i]))
        iteration.append(np.zeros(user_per_sbc[i]))
    user_num = sum(user_per_sbc)
    T_u=0
    for i in range(subcarrier_num):
        for j in range(user_per_sbc[i]):
            uplink_time[i][j]=(parameters*user_per_sbc[i])/((Bandwidth/subcarrier_num)*np.log2(1+(h[i][j]*p_max)))
            iteration[i][j]=(time_of_a_round-uplink_time[i][j])*FLOPS[i][j]/(FLOPs*dataset[i][j])
            if iteration[i][j]<0:
                iteration[i][j]=0
            T_u = T_u + uplink_time[i][j] * FLOPS[i][j]
    # print('TDMA: iteration')
    iteration_flat = None
    for i in range(subcarrier_num):
        iteration_flat = iteration[i] if iteration_flat is None else np.concatenate((iteration_flat, iteration[i]),
                                                                                    axis=0)
    # print(iteration_flat)
    # print(T/(FLOPs * sum_dataset))
    return T_u/(FLOPs * sum_dataset),iteration_flat
def compute(subcarrier_num,user_num):
    time_of_a_round = 5 #second
    P_max = 46 #46 dBm
    Bandwidth = 30 #MHz
    FLOPs = 0.02 #GFLOPs
    parameters = 1.21 * 32 #byte
    hmax = 15 #dB
    hmin = 7 #dB
    Hmax = 10 ** (hmax / 10)
    Hmin = 10 ** (hmin / 10)
    h=[]
    FLOPS=[] #80-200 GFLOPS

    quotient = int(user_num / subcarrier_num) #商
    remainder = user_num - quotient * subcarrier_num#余数
    dataset_size=[] #300, 500 samples
    sum_dataset = 0
    for subc in range(subcarrier_num):
        if subc < remainder:
            h.append(np.zeros(quotient + 1))
            FLOPS.append(10 + 20 * np.random.rand(quotient + 1))
            dataset_size.append(np.random.randint(500, 600,(quotient + 1)))
            sum_dataset+=sum(dataset_size[subc])
        else:
            h.append(np.zeros(quotient))
            FLOPS.append(10 + 20 * np.random.rand(quotient))
            dataset_size.append(np.random.randint(500, 600, quotient))
            sum_dataset += sum(dataset_size[subc])

    print(dataset_size)
    # print('sum_dataset')
    # print(sum_dataset)

    # H = Hmin + Hmax * np.random.rayleigh(size=user_num)
    H = hmin + hmax * np.random.rayleigh(size=user_num)
    # H = Hmin + Hmax * np.random.poisson(size=user_num)

    # H = np.random.triangular(left=hmin,mode=8,right=hmax,size=user_num)
    H.sort()
    count=0
    sum_FLOPS = 0
    for i in range(subcarrier_num):
        for j in range(len(h[i])):
            # h[i][j] = H[j * subcarrier_num + i]
            h[i][j] = 10 ** (H[j * subcarrier_num + i] / 10)
            sum_FLOPS=sum_FLOPS+FLOPS[i][j]
        count=count+len(h[i])
    # print(h)
    # print(FLOPS)

    U_all = sum_FLOPS*time_of_a_round/(FLOPs*sum_dataset)
    # start = time.time()
    opt = resource_allocation.resource_allocate_opt(Bandwidth, FLOPs, sum_dataset, parameters, h, FLOPS,p_max=P_max)



    # first_time = time.time()
    ineq = resource_allocation.resource_allocate_ineq(Bandwidth, FLOPs, sum_dataset, parameters, h, FLOPS,p_max=P_max)
    # end_time = time.time()
    power_only = resource_allocation.power_allocate_opt(Bandwidth, FLOPs, sum_dataset, parameters, h, FLOPS,p_max=P_max)
    tdma,tdma_iteration = TDMA(Bandwidth, FLOPs, sum_dataset, parameters, h, FLOPS, time_of_a_round,dataset_size,p_max=P_max)
    power_max,power_max_iteration = Power_max(Bandwidth, FLOPs, sum_dataset, parameters, h, FLOPS, time_of_a_round,dataset_size, p_max=P_max)
    # uniform_dist,uniform_dist_iteration=Uniform_distribution(Bandwidth, FLOPs, sum_dataset, parameters, h, FLOPS,time_of_a_round,dataset_size,  p_max=P_max)
    # print('power_only_x')
    power_only_x=np.array(power_only.x.tolist()+[Bandwidth/subcarrier_num]*subcarrier_num)
    # print('opt_iteration')
    opt_iteration=get_iteration(opt.x,time_of_a_round,dataset_size,Bandwidth, FLOPs, parameters, h, FLOPS)
    poweronly_iteration=get_iteration(power_only_x,time_of_a_round,dataset_size,Bandwidth, FLOPs, parameters, h, FLOPS)
    print("opt:", U_all - opt.fun, " poweronly:", U_all - power_only.fun, " ineq:", U_all - ineq.fun, " tdma:",
          U_all - tdma, " power_max:", U_all - power_max, " opt/ineq",
          (U_all - opt.fun) / (U_all - ineq.fun))
    return opt_iteration,poweronly_iteration,tdma_iteration,power_max_iteration

    # return user_num, subcarrier_num, U_all-opt.fun,U_all-power_only.fun, U_all-ineq.fun, U_all-tdma,U_all-power_max
def TDMA_asynchronous(Bandwidth, FLOPs, sum_dataset, parameters, h, FLOPS,dataset, p_max=40):
    subcarrier_num = h.__len__()
    iteration_of_around=4
    user_per_sbc,uplink_time,iteration = [],[],[]
    for i in range(subcarrier_num):
        user_per_sbc.append(np.shape(h[i])[0])
        uplink_time.append(np.zeros(user_per_sbc[i]))
        iteration.append(np.zeros(user_per_sbc[i]))
    user_num = sum(user_per_sbc)
    T_u=0
    time_of_a_round=0
    com_time_of_a_round = 0
    # print(FLOPS)
    for i in range(subcarrier_num):
        for j in range(user_per_sbc[i]):
            tij=iteration_of_around *  (FLOPs * dataset[i][j])/FLOPS[i][j]
            # print(i,j)
            # print(tij)
            uplink_time[i][j] = (parameters * user_per_sbc[i]) / (
                        (Bandwidth / subcarrier_num) * np.log2(1 + (h[i][j] * p_max)))
            if time_of_a_round<tij:
                time_of_a_round=tij
            if com_time_of_a_round < uplink_time[i][j]:
                com_time_of_a_round = uplink_time[i][j]
    # time_of_a_round+=com_time_of_a_round

    for i in range(subcarrier_num):
        for j in range(user_per_sbc[i]):
            iteration[i][j] = (time_of_a_round) * FLOPS[i][j] / (FLOPs * dataset[i][j])
            # iteration[i][j]=(time_of_a_round-uplink_time[i][j])*FLOPS[i][j]/(FLOPs*dataset[i][j])
            iteration[i][j]=int((iteration[i][j]+0.1)/(iteration_of_around))*(iteration_of_around)
            T_u = T_u + uplink_time[i][j] * FLOPS[i][j]
    # print('TDMA: iteration')
    iteration_flat = None
    for i in range(subcarrier_num):
        iteration_flat = iteration[i] if iteration_flat is None else np.concatenate((iteration_flat, iteration[i]),
                                                                                    axis=0)
    # print(iteration_flat)
    # print(T/(FLOPs * sum_dataset))
    # print(time_of_a_round)
    # print(com_time_of_a_round)
    total_time=time_of_a_round + com_time_of_a_round
    print(total_time)
    # logging.info("training time of a round")
    logging.info('training time of a round: '+str(time_of_a_round + com_time_of_a_round))
    return T_u/(FLOPs * sum_dataset),iteration_flat#time_of_a_round+com_time_of_a_round
def TDMA_asynchronous2(Bandwidth, FLOPs, sum_dataset, parameters, h, FLOPS,dataset, p_max=40):
    subcarrier_num = h.__len__()
    iteration_of_around = 12
    user_per_sbc,uplink_time,iteration, interation_Asynchronous = [],[],[],[]
    for i in range(subcarrier_num):
        user_per_sbc.append(np.shape(h[i])[0])
        uplink_time.append(np.zeros(user_per_sbc[i]))
        iteration.append(np.zeros(user_per_sbc[i]))
        interation_Asynchronous.append(np.zeros(user_per_sbc[i]))
    user_num = sum(user_per_sbc)
    T_u=0
    time_of_a_round=0
    com_time_of_a_round = 0
    # print(FLOPS)
    for i in range(subcarrier_num):
        for j in range(user_per_sbc[i]):
            tij=iteration_of_around *  (FLOPs * dataset[i][j])/FLOPS[i][j]
            # print(i,j)
            # print(tij)
            uplink_time[i][j] = (parameters * user_per_sbc[i]) / (
                        (Bandwidth / subcarrier_num) * np.log2(1 + (h[i][j] * p_max)))
            if time_of_a_round<tij:
                time_of_a_round=tij
            if com_time_of_a_round < uplink_time[i][j]:
                com_time_of_a_round = uplink_time[i][j]
    # time_of_a_round+=com_time_of_a_round
    max_rounds=0
    for i in range(subcarrier_num):
        for j in range(user_per_sbc[i]):
            iteration[i][j] = (time_of_a_round) * FLOPS[i][j] / (FLOPs * dataset[i][j])
            # iteration[i][j]=(time_of_a_round-uplink_time[i][j])*FLOPS[i][j]/(FLOPs*dataset[i][j])
            if int((iteration[i][j]+0.1)/(iteration_of_around))>max_rounds:
                max_rounds=int((iteration[i][j] + 0.1) / (iteration_of_around))

    for i in range(subcarrier_num):
        for j in range(user_per_sbc[i]):
            interation_Asynchronous[i][j] = (time_of_a_round+com_time_of_a_round*(max_rounds-int((iteration[i][j] + 0.1) / (iteration_of_around)))) * FLOPS[i][j] / (FLOPs * dataset[i][j])
            # iteration[i][j]=(time_of_a_round-uplink_time[i][j])*FLOPS[i][j]/(FLOPs*dataset[i][j]
            iteration[i][j] = int((iteration[i][j] + 0.1) / (iteration_of_around)) * (iteration_of_around)
            interation_Asynchronous[i][j]=int((interation_Asynchronous[i][j]+0.1)/(iteration_of_around))*(iteration_of_around)
            T_u = T_u + uplink_time[i][j] * FLOPS[i][j]
    # print('TDMA: iteration')
    iteration_flat = None
    for i in range(subcarrier_num):
        iteration_flat = interation_Asynchronous[i] if iteration_flat is None else np.concatenate((iteration_flat, interation_Asynchronous[i]),
                                                                                    axis=0)
    # print(iteration_flat)
    # print(T/(FLOPs * sum_dataset))
    print(time_of_a_round)
    print(com_time_of_a_round)
    print(time_of_a_round + max_rounds*com_time_of_a_round)
    print(iteration)
    print(interation_Asynchronous)
    return T_u/(FLOPs * sum_dataset),iteration_flat#time_of_a_round+com_time_of_a_round
def iteration_out(usernum,sbc):
    compute_in = compute(sbc, usernum)
    # print(len(compute_in))
    for i in range(len(compute_in)):
        print(compute_in[i])
    filename = 'iteration/iteration_'+'user_'+str(usernum)+'_sbc_'+str(sbc)+'.npz'
    np.savez(filename,opt_iteration=np.array(compute_in[0]), poweronly_iteration=np.array(compute_in[1]), tdma_iteration=np.array(compute_in[2]),power_max_iteration=np.array(compute_in[3]))
def unopersbc_repeattimes(unum,repeat,sbc):
    repeat_times = repeat

    com = []
    positions_box = []
    # time_opt_box = []
    # time_ineq_box = []
    U_opt = []
    U_ineq = []
    U_TDMA = []
    U_Power = []
    U_uniform_dist = []
    U_Power_max = []
    # for usernum in range(2*sbc,unum):
    for usernum in range(unum-1, unum):
        positions_box.append(usernum)
        # print('positions_box'+str(sbc_num * (user_num_per_subcarrier + 2)))
        # time_opt = []
        # time_ineq = []
        Uopt = 0
        Upower=0
        Upowermax = 0
        Uineq = 0
        Utdma = 0
        Uuniform_dist = 0
        repeat_times_a=repeat_times
        for r in range(repeat_times):
            print(r,usernum)
            compute_in = compute(sbc, usernum)
            if compute_in[4]>60:
                repeat_times_a-=1
                continue
            elif compute_in[4]<0:
                repeat_times_a -= 1
                continue
            else:
                com.append(compute_in)
                # time_opt.append(compute_in[5])
                # time_ineq.append(compute_in[6])
                Uopt = Uopt + compute_in[2]
                Upower = Upower + compute_in[3]
                Uineq = Uineq + compute_in[4]
                Utdma = Utdma + compute_in[5]
                Uuniform_dist = Uuniform_dist + compute_in[6]
                Upowermax = Upowermax+compute_in[7]
        U_opt.append(Uopt / repeat_times_a)
        U_ineq.append(Uineq / repeat_times_a)
        U_Power.append(Upower / repeat_times_a)
        U_TDMA.append(Utdma / repeat_times_a)
        U_uniform_dist.append(Uuniform_dist/repeat_times_a)
        U_Power_max.append(Upowermax/repeat_times_a)
        # time_opt_box.append(time_opt)
        # time_ineq_box.append(time_ineq)

    c = np.array(com)
    # print(c[:, 2])
    # print(c[:, 3])
    # print(c[:, 6])
    # box_t_ineq = pd.DataFrame({'User Number': c[:, 0], "Compute Time (s)": c[:, 5], 'Method': ineq_label})
    # box_t_opt = pd.DataFrame({'User Number': c[:, 0], "Compute Time (s)": c[:, 4], 'Method': opt_label})
    # box_data = pd.concat([box_t_ineq, box_t_opt])
    # print(box_data)
    filename='max_iteration/U_car_'+str(sbc)+'_user_'+str(unum)+'_repeat_'+str(repeat_times)+'.npz'
    np.savez(filename, doc_c=c,doc_test_times=repeat_times, doc_positions_box=np.array(positions_box),
             doc_U_opt=np.array(U_opt), doc_U_power=np.array(U_Power), doc_U_ineq=np.array(U_ineq),doc_U_TDMA=np.array(U_TDMA),doc_U_uniform_dist=np.array(U_uniform_dist),doc_U_Power_max_dist=np.array(U_Power_max))
def get_iteration(x,time_of_a_round,dataset,Bandwidth, FLOPs, parameters, h, FLOPS):
    # print(h)
    subcarrier_num = h.__len__()
    user_per_sbc = []
    for i in range(subcarrier_num):
        user_per_sbc.append(np.shape(h[i])[0])
    user_num = sum(user_per_sbc)

    subcarrier_bandwidth = x[user_num:]
    power_flat = x[:user_num]
    power, A, y, uplink_time, iteration, pointer = [], [], [], [], [], 0
    for i in range(subcarrier_num):
        power.append (power_flat[pointer:pointer + user_per_sbc[i]])
        pointer += user_per_sbc[i]
        A.append(h[i] * power[-1])
        y.append(np.zeros(user_per_sbc[i], dtype=float))
        uplink_time.append(np.zeros(user_per_sbc[i], dtype=float))
        iteration.append(np.zeros(user_per_sbc[i], dtype=float))
    # A = h * power
    # y = np.zeros((subcarrier_num, int(user_num / subcarrier_num)))
    # uplink_time = np.zeros((subcarrier_num, int(user_num / subcarrier_num)))
    for i in range ( subcarrier_num ):
        for j in range(user_per_sbc[i]):
            # y[i, j] = A[i, j] + y[i, j - 1]
            y[i][j] = A[i][j] + y[i][j - 1]
    for i in range(subcarrier_num):
        for j in range(user_per_sbc[i]):
            # y[i, j] = subcarrier_bandwidth[i] * subcarrier_num / Bandwidth + y[i, j]
            y[i][j] = subcarrier_bandwidth[i] * subcarrier_num / Bandwidth + y[i][j]
    for i in range(subcarrier_num):
        for j in range(user_per_sbc[i]):
            uplink_time[i][j] = parameters / (subcarrier_bandwidth[i] * (np.log2(y[i][j] / (y[i][j] - A[i][j]))))
    # cvx_U_uplink = sum(sum(FLOPS * uplink_time)) / (FLOPs * sum_dataset)
    # cvx_U_uplink = 0
    # print(user_per_sbc)
    for i in range(subcarrier_num):
        for j in range(user_per_sbc[i]):
            # print(i,j)
            # print(dataset[i][j])
            iteration[i][j]=(time_of_a_round-uplink_time[i][j])*FLOPS[i][j]/(FLOPs*dataset[i][j])
            if iteration[i][j]<1:
                iteration[i][j]=1
    # print('iteration')
    iteration_flat = None
    for i in range(subcarrier_num):
        iteration_flat = iteration[i] if iteration_flat is None else np.concatenate((iteration_flat, iteration[i]), axis=0)
    # print(iteration_flat)

    return iteration_flat
def iteration_api(round,user_num,subcarrier_num, datasize,com_mode):
    time_of_a_round = 41  # second
    P_max = 46  # 46 dBm
    Bandwidth = 30  # MHz

    # FLOPs = 0.02  # GFLOPs
    # parameters = 1.21 * 32  # byte
    FLOPs = 0.04  # GFLOPs
    parameters = 11.69 * 32  # byte
    hmax = 8  # dB
    hmin = 7  # dB
    Hmax = 10 ** (hmax / 10)
    Hmin = 10 ** (hmin / 10)
    h = []
    FLOPS = []  # 80-200 GFLOPS

    quotient = int(user_num / subcarrier_num)  # 商
    remainder = user_num - quotient * subcarrier_num  # 余数
    dataset_size = []  # 300, 500 samples
    sum_dataset = 0
    for subc in range(subcarrier_num):
        if subc < remainder:
            h.append(np.zeros(quotient + 1))
            np.random.seed(subc*round+1)
            FLOPS.append(6 + 3 * np.random.rand(quotient + 1))
            dataset_size.append(datasize[subc*(quotient+1):(subc+1)*(quotient+1)])
            # print(subc)
            # print(dataset_size[subc])
            sum_dataset += sum(dataset_size[subc])
        else:
            h.append(np.zeros(quotient))
            np.random.seed(subc*round+2)
            FLOPS.append(6 + 3* np.random.rand(quotient))
            dataset_size.append(datasize[subc*quotient+remainder:(subc+1)*quotient+remainder])
            sum_dataset += sum(dataset_size[subc])

    # print(dataset_size)
    # print('sum_dataset')
    # print(sum_dataset)

    # H = Hmin + Hmax * np.random.rayleigh(size=user_num)
    np.random.seed(round+5)
    H = hmin + hmax * np.random.rayleigh(size=user_num)
    # H = Hmin + Hmax * np.random.poisson(size=user_num)

    # H = np.random.triangular(left=hmin,mode=8,right=hmax,size=user_num)
    H.sort()
    count = 0
    sum_FLOPS = 0
    for i in range(subcarrier_num):
        for j in range(len(h[i])):
            # h[i][j] = H[j * subcarrier_num + i]
            h[i][j] = 10 ** (H[j * subcarrier_num + i] / 10)
            sum_FLOPS = sum_FLOPS + FLOPS[i][j]
        count = count + len(h[i])
    # iteration_of_around=10
    out_iteration=None
    if com_mode==0:
        opt = resource_allocation.resource_allocate_opt(Bandwidth, FLOPs, sum_dataset, parameters, h, FLOPS,
                                                        p_max=P_max)
        opt_iteration = get_iteration(opt.x, time_of_a_round, dataset_size, Bandwidth, FLOPs, parameters, h, FLOPS)
        out_iteration=opt_iteration
    elif com_mode==1:
        power_only = resource_allocation.power_allocate_opt(Bandwidth, FLOPs, sum_dataset, parameters, h, FLOPS,
                                                            p_max=P_max)
        power_only_x = np.array(power_only.x.tolist() + [Bandwidth / subcarrier_num] * subcarrier_num)
        poweronly_iteration = get_iteration(power_only_x, time_of_a_round, dataset_size, Bandwidth, FLOPs, parameters,
                                            h, FLOPS)
        out_iteration=poweronly_iteration
    elif com_mode==2:
        tdma, tdma_iteration = TDMA(Bandwidth, FLOPs, sum_dataset, parameters, h, FLOPS, time_of_a_round, dataset_size,
                                    p_max=P_max)
        out_iteration=tdma_iteration
    elif com_mode==3:
        power_max, power_max_iteration = Power_max(Bandwidth, FLOPs, sum_dataset, parameters, h, FLOPS, time_of_a_round,
                                                   dataset_size, p_max=P_max)
        out_iteration=power_max_iteration
    elif com_mode==4:
        tdma2, tdma_iteration2 = TDMA_asynchronous(Bandwidth, FLOPs, sum_dataset, parameters, h, FLOPS, dataset_size,
                                    p_max=P_max)
        out_iteration=tdma_iteration2
    elif com_mode == 5:
        tdma2, tdma_iteration3 = TDMA_asynchronous2(Bandwidth, FLOPs, sum_dataset, parameters, h, FLOPS,
                                                    dataset_size,
                                                   p_max=P_max)
        out_iteration = tdma_iteration3
    # print(com_mode,out_iteration)
    return out_iteration
if __name__ == "__main__":
    repeat_times = 5
    usernumbers = 25
    sbcnum = 10


    # iteration_out(usernumbers,sbcnum)
    # unopersbc_repeattimes(usernumbers,repeat_times,sbcnum)

    # quotient = int(usernumbers / sbcnum)  # 商
    # remainder = usernumbers - quotient * sbcnum  # 余数
    datasetsize = 100*np.ones (usernumbers)  # 300, 500 samples
    # datasetsize =   np.random.randint(250,500,usernumbers)
    # # sum_dataset = 0
    # datasetsize=[]
    # for subc in range(sbcnum):
    #     if subc < remainder:
    #         np.random.seed(round + 1)
    #         datasetsize.append(100*np.ones (quotient + 1))
    #     else:
    #         np.random.seed(round + 2)
    #         datasetsize.append(100*np.ones (quotient))


    print(datasetsize)
    for r in range(1000):
        a=iteration_api(r,usernumbers,sbcnum,datasetsize,2)
        print(min(a))
        # print(sum(a)/usernumbers)



