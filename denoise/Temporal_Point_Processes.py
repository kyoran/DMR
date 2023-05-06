import numpy as np

def HP(events, u, w, mx, my, flow, t_ref, min_i, a=np.array([[1.0, 0.0], [0.0, 1.0]]), rangeX=240, rangeY=180):
    '''

    Using Hawkes Process to up-sampling events by Ogata's thinning algorithm.

    :input: sparse events

    :param u: intensity of current trajectory
    :param w: basic intensity
    (u = w * events number)
    :param a: marked point process, on event can only affect the generation of on event
    :param mx, my: x, y coordinate of the current main event trajectory
    :param flow: the direction of the current trajectory
    :param t_ref: reference time plane, for calculating the x, y coordinate of the up-sampling events
    :param min_i: the minimum time interval of the current trajectory
    :param rangeX, rangeY: x, y range of input events

    :return: up-sampling events

    '''

    # initialize current time and current intensity
    s = events[0, -1]
    cur_intensity = np.sum(u)

    # record up-sampling events, including original input events
    gen_events = []
    gen_events.append(events[0, :])
    cnt = 1
    next_event = events[cnt, :]

    last_intensity = u.copy()
    reject_event = False

    while True:
        # timestamp and polarity of the last event
        tj, pj = gen_events[-1][-1], int(gen_events[-1][-2])

        if reject_event:
            cur_intensity = np.sum(intensity)
            reject_event = False
        else:
            cur_intensity = np.sum(last_intensity) + w * np.sum(a[:, pj])

        # generate new time interval
        cur_s = np.random.exponential(scale=1. / cur_intensity)
        s += cur_s

        # calculate intensity at time s
        intensity = u + np.exp(-w * (s - tj)) * (a[:, pj].flatten() * w + last_intensity - u)

        # adding the next original event
        if s > next_event[-1]:
            s = next_event[-1]
            gen_events.append(next_event)
            if cnt < events.shape[0] - 1:
                cnt += 1
                next_event = events[cnt, :]
                continue
            else:
                gen_events = np.array(gen_events)
                return gen_events

        # limit the minimum time interval for the up-sampling events
        if cur_s < min_i or np.abs(next_event[-1] - s) < min_i:
            decIstar = True
            continue

        # thinning process
        thin = cur_intensity - np.sum(intensity) # If diff is not 0, there is a certain probability that the newly generated points will not be retained
        try:
            n0 = np.random.choice(np.arange(len(u) + 1), 1,
                                  p=(np.append(intensity, thin) / cur_intensity))
        except ValueError:
            print('Probabilities do not sum to one.')
            gen_events = np.array(gen_events)
            return gen_events

        # save the up-sampling event
        if n0 < len(u):
            gen_event_x = round(mx - flow[0] * (t_ref - s))
            gen_event_y = round(my - flow[1] * (t_ref - s))
            if gen_event_x >= 0 and gen_event_x < rangeX and gen_event_y >= 0 and gen_event_y < rangeY:
                gen_events.append([gen_event_x, gen_event_y, n0[0], s])
                lastrates = intensity.copy()
            else:
                decIstar = True
        else:
            decIstar = True

def SP(events, t_min, t_max, u, b, th, mx, my, flow, rangeX, rangeY):
    '''

    Up-sampling events by Self-correcting Process

    :input: sparse events

    :param t_min: minimum timestamp
    :param t_max: maximum timestamp
    :param u: basic intensity
    :param b: drop weight
    :param th: events generating threshold

    :return: up-sampling events
    '''
    gen_events = []
    gen_events.append(events)
    cur_t = np.random.randint(events[3], t_max)
    while cur_t < t_max:
        intensity = np.exp((cur_t-t_min) * u - b)
        if intensity > th:
            gen_event_x = round(mx - flow[0] * (t_max - cur_t))
            gen_event_y = round(my - flow[1] * (t_max - cur_t))
            if gen_event_x >= 0 and gen_event_x < rangeX and gen_event_y >= 0 and gen_event_y < rangeY:
                gen_events.append([gen_event_x, gen_event_y, events[2], cur_t])
        else:
            cur_t += np.random.randint(cur_t, t_max+1)
    return np.array(gen_events)
