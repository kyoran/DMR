# !/usr/bin/python3
# -*- coding: utf-8 -*-
# Created by kyoRan on 2022/10/16 10:44

import shutil
import sys
import os

def make_dir(dir_path, check=False):
    try:
        print(dir_path)
        if check:
            try:
                assert not os.path.exists(dir_path), 'specified working directory already exists'
            except:
                # is_delete = input("are you ready to delete existing logs (y/n): ")
                # if is_delete == 'y':
                shutil.rmtree(dir_path)
                # else:
                #     sys.exit(1)


        os.mkdir(dir_path)


    except OSError:
        pass

    return dir_path