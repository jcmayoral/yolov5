import argparse
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import pandas as pd

def process(opt, normalize=1):
    samples = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    stats = ['P', 'R', 'mAP@.5', 'mAP@.5:.95']
    classes = ['all', 'Lethal', 'Danger', 'Warning', "Safe"]

    #linspace for confidence threshold
    for ind,s in enumerate(samples):
        #container = DataContainer()
        #one for stats
        fig, axs = plt.subplots(5,5)
        fig.suptitle("Confusion Matrix with Confidence Threshold set to  {}".format(s))

        data = np.zeros((11, 5, 5))

        #look for files and read if matches with pattern
        for files in glob.glob(os.path.join(opt.folder,"cm_"+str(s)+"*.txt")):
            cm = np.loadtxt(files, dtype=np.float, delimiter=' ')
            #Get confidence form file results_<conf>_<iou>
            #conf_thr == s
            fileName, _ = os.path.splitext(files)

            conf_thr = fileName.split('_')[1]
            iou_thr = fileName.split('_')[2]

            #index from iou_thr
            n = int(float(iou_thr)*10)

            cm = cm / ((cm.sum(0).reshape(1, -1) + 1E-9) if normalize else 1)  # normalize columns


            data[n] = cm

        for i in range(5):
            for j in range(5):
                axs[i,j].plot(samples,  data[:,i,j], label="CM Index {} {}".format(i,j))

        for i in range(25):
            r = i//5
            c = i%5
            axs[c,r].set_title("CM Index {} {}".format(i,j))
            #axs[c,r].legend()
            axs[c,r].grid()
            axs[c,r].set_xlabel('IoU threshold')

        plt.legend()
        plt.show()

def parse_opt():
    #parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, default='data', help='folder cointaining txt')
    #parser.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold')
    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    opt = parse_opt()
    process(opt)
