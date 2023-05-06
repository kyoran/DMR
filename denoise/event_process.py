import numpy as np
import matplotlib.pyplot as plt

# event trajectory along the flow direction
def event_trace(flow, x, y, p, t, t_ref, rangeX, rangeY):
    trace = [[[] for ix in range(rangeY)] for iy in range(rangeX)]
    for i in range(x.size):
        cur_x = round(x[i] + flow[0] * (t_ref - t[i]))
        cur_y = round(y[i] + flow[1] * (t_ref - t[i]))
        if cur_x >= 0 and cur_x < rangeX and cur_y >= 0 and cur_y < rangeY:
            trace[int(cur_x)][int(cur_y)].append([x[i], y[i], p[i], t[i]])
    return trace

# project events along the flow onto the reference time plane t_ref
def warp(flow, x, y, p, t, t_ref, rangeX, rangeY):
    ref = np.zeros((rangeX, rangeY)).astype(np.int)
    for i in range(x.size):
        cur_x = round(x[i] + flow[0] * (t_ref - t[i]))
        cur_y = round(y[i] + flow[1] * (t_ref - t[i]))
        if cur_x >= 0 and cur_x < rangeX and cur_y >= 0 and cur_y < rangeY:
            if p[i] > 0:
                ref[int(cur_x), int(cur_y)] += 1
            else:
                ref[int(cur_x), int(cur_y)] -= 1
    return ref

# distinguish main events and noise
def dist_main_noise(flow, x, y, p, t, t_ref, rangeX, rangeY):
    ref = warp(flow, x, y, p, t, t_ref, rangeX, rangeY)
    nx, ny = np.where(np.abs(ref) == 1)  # noises
    mx, my = np.where(np.abs(ref) >= 2)  # main events
    return ref, nx, ny, mx, my

def min_inter(t_ref, mx, my, trace):
    min_i = t_ref
    for i in range(mx.size):
        cur_events = np.array(trace[mx[i]][my[i]])
        diff = cur_events[1:, -1] - cur_events[:-1, -1]
        min_diff = np.min(diff)
        if min_diff < min_i:
            min_i = min_diff
    return min_i

# limit the number of events generated, not more than r times of the event number in current trajectory
def limit_num(ori_events, gen_events, limit_up_rate):
    if gen_events.shape[0] > ori_events.shape[0] * limit_up_rate:
        select_events = ori_events
        out_t = np.setdiff1d(gen_events[:, -1], ori_events[:, -1])
        select_t = np.random.choice(out_t.size, ori_events.shape[0] * (limit_up_rate - 1), replace=False)
        for j in range(select_t.size):
            idx = np.where(gen_events[:, -1] == out_t[select_t[j]])[0]
            select_events = np.append(select_events, gen_events[idx, :], axis=0)
        return select_events
    else:
        return gen_events

def show_ref(params, x, y, p, t, t_ref, rangeX, rangeY):
    ref = np.zeros((rangeX, rangeY))
    for i in range(x.size):
        
        # without polarity
        # cur_x = round(x[i] + -1.72707543e-07 * (t_ref - t[i]))
        # cur_y = round(y[i] + 1.15075271e-04 * (t_ref - t[i]))
        
        # with polarity
        cur_x = round(x[i] + params[0] * (t_ref - t[i]))
        cur_y = round(y[i] + params[1] * (t_ref - t[i]))
        # cur_x = x[i]
        # cur_y = y[i]
        if cur_x >= 0 and cur_x < rangeX and cur_y >= 0 and cur_y < rangeY:
            if p[i] > 0 :
                ref[cur_x, cur_y] += 1
            else:
                ref[cur_x, cur_y] -= 1

    print(np.var(ref))
    # plt.matshow(ref, cmap=plt.cm.RdBu)
    # plt.colorbar()
    # plt.show()

def show_events(gen_events, x, y, p, t, ori=True, gen=True):

    if ori:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        on_idx = np.where(p==True)
        off_idx = np.where(p==False)
        ax.scatter(x[on_idx], t[on_idx], y[on_idx], s=3, c='r')
        ax.scatter(x[off_idx], t[off_idx], y[off_idx], s=3, c='b')
        plt.show()

    if gen:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        gen_on_idx = np.where(gen_events[:, 2]==True)
        gen_off_idx = np.where(gen_events[:, 2]==False)
        ax.scatter(gen_events[gen_on_idx, 0], gen_events[gen_on_idx, 3], gen_events[gen_on_idx, 1], s=3, c='g')
        ax.scatter(gen_events[gen_off_idx, 0], gen_events[gen_off_idx, 3], gen_events[gen_off_idx, 1], s=3, c='y')
        plt.show()


def save_event(gen_events, x, y, p, t, ori=True, gen=True, save_path='results/'):

    if ori:
        fw = open(save_path + "ori.txt", 'w')
        for line in range(len(x)):
            fw.write('{} {} {} {}'.format(t[line], x[line], y[line], int(p[line])))
            fw.write("\n")
        fw.close()

    if gen:
        all = gen_events[gen_events[:, -1].argsort()]
        fw = open(save_path + "up.txt", 'w')
        for line in range(all.shape[0]):
            fw.write('{} {} {} {}'.format(all[line, -1], all[line, 0], all[line, 1], all[line, 2]))
            fw.write("\n")
        fw.close()
