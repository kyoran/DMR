# !/usr/bin/python3
# -*- coding: utf-8 -*-
# Created by kyoRan on 2022/10/18 11:46

def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(
            tau * param.data + (1 - tau) * target_param.data
        )