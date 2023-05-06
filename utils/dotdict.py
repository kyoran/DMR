# !/usr/bin/python3
# -*- coding: utf-8 -*-
# Created by kyoRan on 2022/11/10 11:24


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
