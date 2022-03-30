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
        #fig, axs = plt.subplots(5,5)
        fig, axs = plt.subplots(4, sharex=True)
        plt.subplots_adjust(hspace=0.50)

        fig.suptitle("Confusion Matrix with Confidence Threshold set to  {}".format(s))

        non_data = np.zeros((11, 5, 5))
        data = np.zeros((11, 5, 5))

        number_files = 1
        n_images = 1

        #look for files and read if matches with pattern
        for files in glob.glob(os.path.join(opt.folder,"cm_"+str(s)+"*.txt")):
            non_cm = np.loadtxt(files, dtype=np.float, delimiter=' ')
            #Get confidence form file results_<conf>_<iou>
            #conf_thr == s
            fileName, _ = os.path.splitext(files)

            conf_thr = fileName.split('_')[1]
            iou_thr = fileName.split('_')[2]

            #index from iou_thr
            n = int(float(iou_thr)*10)

            cm = non_cm / ((non_cm.sum(0).reshape(1, -1) + 1E-9) if normalize else 1)  # normalize columns

            #Assuming same size of set
            number_files = non_cm.sum(0)
            n_images = number_files[:4].sum()

            data[n] = cm
            non_data[n] = non_cm

        #
        #for i in range(5):
        #    for j in range(5):
        #        axs[i,j].plot(samples,  data[:,i,j], label="CM Index {} {}".format(i,j))

        #Precision
        for j in range(4):
            axs[0].plot(samples,  data[:,j,j], label="Precision {}".format(j))
            axs[0].set_title("Normalize Precision")

        #FP
        for j in range(4):
            axs[1].plot(samples,  data[:,j,-1], label="FP {}".format(j))
            axs[1].set_title("False Positives Percentual")

        #FN
        for j in range(4):
            axs[2].plot(samples,  non_data[:,-1,j], label="FN {}".format(j))
            axs[2].set_title("Number of False Negatives Detections")

        #LETHAL MISCLASSIFICATIONS
        lmisclass = non_data[:,0,1] + non_data[:,0,2] + non_data[:,0,3] + non_data[:,1,0] + non_data[:,2,0] + non_data[:,3,0]
        axs[3].plot(samples, lmisclass/number_files[0] , label="Lethal")
        #0,1 0,2 0,3
        #1,0 2,0 3,0
        axs[3].set_title("Misclassifications")

        #Danger MISCLASSIFICATIONS
        dmisclass = non_data[:,0,1] + non_data[:,2,1] + non_data[:,3,1] + non_data[:,1,0] + non_data[:,1,2] + non_data[:,1,3]
        axs[3].plot(samples, dmisclass/number_files[1] , label="Danger")
        #0,1 2,1 3,1
        #1,0 1,2 1,3

        #Warning MISCLASSIFICATIONS
        wmisclass = non_data[:,2,0] + non_data[:,2,1] + non_data[:,2,3] + non_data[:,0,2] + non_data[:,1,2] + non_data[:,3,2]
        axs[3].plot(samples, wmisclass/number_files[2] , label="Warning")
        #2,0 2,1 2,3
        #0,2 1,2 3,2

        #Safe MISCLASSIFICATIONS
        smisclass = non_data[:,3,0] + non_data[:,3,1] + non_data[:,3,2] + non_data[:,0,3] + non_data[:,1,3] + non_data[:,2,3]
        axs[3].plot(samples, smisclass/number_files[3] , label="Safe")
        #3,0 3,1 3,2
        #0,3 1,3 2,3

        #all
        all_misclass = non_data[:,0,1] + non_data[:,0,2] + non_data[:,0,3] \
                     + non_data[:,1,0] + non_data[:,1,2] + non_data[:,1,3] \
                     + non_data[:,2,0] + non_data[:,2,1] + non_data[:,2,3] \
                     + non_data[:,3,0] + non_data[:,3,1] + non_data[:,3,2]

        axs[3].plot(samples, all_misclass/n_images , label="ALL")

        ticks = np.arange(0, 1.1, 0.1)
        y_ticks = np.arange(0, 1.1, 0.2)


        for i in range(4):
            box = axs[i].get_position()
            axs[i].set_position([box.x0, box.y0, box.width * 0.8, box.height])
            axs[i].legend(loc='center left', bbox_to_anchor=(1, 0.5))
            axs[i].grid()
            axs[i].set_xticks(ticks)
            if i == 2:
                continue
            axs[i].set_yticks(y_ticks)

        axs[3].set_xlabel('IoU threshold')


        #for i in range(25):
        #    r = i//5
        #    c = i%5
        #    axs[c,r].set_title("CM Index {} {}".format(i,j))
        #    #axs[c,r].legend()
        #    axs[c,r].grid()
        #    axs[c,r].set_xlabel('IoU threshold')

        plt.legend()
        image_filename = os.path.join(opt.folder,"results_"+str(s)+"*.jpg")
        #plt.show()
        plt.savefig(image_filename, dpi=500)

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
