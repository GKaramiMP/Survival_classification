from matplotlib.gridspec import GridSpec
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from matplotlib.font_manager import FontProperties
from matplotlib import colors as mcolors
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)

def smooth_curve(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth
if __name__=='__main__':
    fontP = FontProperties()
    fontP.set_size('small')

    log_name_train_type = {'Training': 'train/',
                                   'Validation': 'validation/',
                                   'Test': 'test/',
                           }
    train_mode_list = ['Training','Validation','Test']#,
    tag_list = ['accuracy_survival', 'cross_entropy','accuracy_grade' ]
    accuracy_survival = []
    accuracy_survival_vl = []
    accuracy_survival_ts = []
    cross_entropy = []
    cross_entropy_vl = []
    cross_entropy_ts = []
    accuracy_grading=[]
    accuracy_grading_vl=[]
    accuracy_grading_ts=[]

    for train_mode in train_mode_list:
        log_folder='/exports/lkeb-hpc/gkarami/Code/Log/34/'
        log_test_folder = log_folder + log_name_train_type[train_mode]

        loss_dict = dict()
        file_list = [f for f in os.listdir(log_test_folder) if os.path.isfile(os.path.join(log_test_folder, f))]

        for file in file_list:


            for e in tf.train.summary_iterator(log_test_folder + file):
                broken_point= 300000 #27725 for shark8, 30050 for shark7
                if e.step==broken_point:
                    break
                for v in e.summary.value:
                    # print(v.tag)
                    if train_mode=='Training':
                        if v.tag == tag_list[0]:
                            accuracy_survival.append(v.simple_value )
                        elif v.tag == tag_list[1]:
                            cross_entropy.append(v.simple_value)
                        elif v.tag == tag_list[2]:
                            accuracy_grading.append(v.simple_value)

                    if train_mode == 'Validation':
                        if v.tag == tag_list[0]:
                            accuracy_survival_vl.append(v.simple_value)
                        elif v.tag == tag_list[1]:
                            cross_entropy_vl.append(v.simple_value)
                        elif v.tag == tag_list[2]:
                            accuracy_grading_vl.append(v.simple_value)
                    if train_mode == 'Test':
                        if v.tag == tag_list[0]:
                            accuracy_survival_ts.append(v.simple_value)
                        elif v.tag == tag_list[1]:
                            cross_entropy_ts.append(v.simple_value)
                        elif v.tag == tag_list[2]:
                            accuracy_grading_ts.append(v.simple_value)
    end_lim=130000
    rng=list(range(0,  len(accuracy_survival[:end_lim])))
    # rng_vl=list(range(0, 125 * len(loss_vl), 125))
    # 8cffdb,137e6d
    #==================================


    fig = plt.figure()

    gs = GridSpec(1, 3)
    ax2 = fig.add_subplot(gs[0, 0])
    plt.plot(rng, smooth_curve(cross_entropy[:end_lim], 20), c='tomato', label='train')
    plt.plot(rng, smooth_curve(cross_entropy_vl[:end_lim], 20), c='lightgreen', label='validation')
    # plt.plot(rng, smooth_curve(cross_entropy_ts[:end_lim], 20), c='#916e99', label='test')
    plt.ylabel('Loss function')
    plt.xlabel('Time point')
    plt.xlim([0, end_lim])
    plt.grid()
    ax2.legend()

    # major_ticks = np.arange(0, end_lim, 10000)
    # minor_ticks = np.arange(0, end_lim, 5000)
    # ax2.set_xticks(major_ticks)
    # ax2.set_xticks(minor_ticks, minor=True)
    # ax2.set_yticks(major_ticks)
    # ax2.set_yticks(minor_ticks, minor=True)
    # ax2.grid(which='both')
    # ax2.grid(which='minor', alpha=0.2)
    # ax2.grid(which='major', alpha=0.5)


    ax1 = fig.add_subplot(gs[0, 1])  # First row, first column
    # plt.plot(rng, smooth_curve(accuracy_survival, 1), c='#fed0fc', alpha=.6)
    plt.plot(rng, smooth_curve(accuracy_survival[:end_lim], 20), c='#c875c4', label='train')
    plt.plot(rng, smooth_curve(accuracy_survival_vl[:end_lim], 20), c='#4b57db', label='validation')
    # plt.plot(rng, smooth_curve(accuracy_survival_ts[:end_lim], 20), c='#916e99', label='test')
    plt.ylabel('Survival accuracy')
    plt.xlabel('Time point')
    plt.xlim([0,end_lim])
    plt.grid()
    ax1.legend()

    ax3 = fig.add_subplot(gs[0, 2])  # First row, first column
    # plt.plot(rng, smooth_curve(accuracy_survival, 1), c='#fed0fc', alpha=.6)
    plt.plot(rng, smooth_curve(accuracy_grading[:end_lim], 20), c='pink', label='train')
    plt.plot(rng, smooth_curve(accuracy_grading_vl[:end_lim], 20), c='violet', label='validation')
    # plt.plot(rng, smooth_curve(accuracy_survival_ts[:end_lim], 20), c='#916e99', label='test')
    plt.ylabel('Grading accuracy')
    plt.xlabel('Time point')
    plt.xlim([0, end_lim])
    plt.grid()
    ax3.legend()

    plt.show()

