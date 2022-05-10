import argparse
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def my_plot(samples,data, image_filename, ylabel):
    plt.figure()
    for i in range(data.shape[0]):
        plt.scatter(samples, data[i,:], label='coeff_{0:.2}'.format(i*0.1))

    plt.xlabel('IOU threshold')
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(image_filename, dpi=1500)

def my_plot2(x,data, image_filename, xlabel, ylabel):
    plt.figure()
    color = get_cmap(11)

    for i in range(data.shape[0]):
        plt.figure()
        plt.scatter(x[i,:], data[i,:], label='coeff_{0:.2}'.format(i*0.1), c=color(i), s=10+20*np.arange(data[i,:].shape[0]), alpha=1.0)
        #plt.scatter(x[i,:], data[i,:], s=50*np.arange(data[i,:].shape[0]), marker="x", facecolor=color(i))
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.savefig(image_filename.replace(".jpg", str(i)+".jpg"), dpi=1500)

def process(opt):
    samples = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]

    acc_results = np.zeros((11,11))
    exp_results = np.zeros((11,11))
    miss_results = np.zeros((11,11))
    coeff_results = np.zeros((11,11))

    #linspace for confidence threshold
    for ind,s in enumerate(samples):
        #look for files and read if matches with pattern
        for files in glob.glob(os.path.join(opt.folder,"cm_"+str(s)+"*.txt")):
            FP = 0
            FN = 0
            TP = 0
            TN = 0
            ACC = 0
            MISCLASSES = 0
            #NUMBER OF LABELS
            TOTAL = 1360

            frame = pd.read_csv(files,delim_whitespace=True, header=None)
            arr = frame.to_numpy()
            #
            TP = arr[:4,:4].diagonal().sum()
            MISCLASSES += np.extract(1 -  np.eye(4), arr[:4,:4]).sum()
            FP2 = arr[:4,-1].sum()
            FN = arr[-1,:4].sum()
            ACC = (TP+TN)/TOTAL#(TP+TN+FP+FN)
            ACC2 = (TP+TN)/(MISCLASSES+TOTAL)
            MISCLASS_RATE = MISCLASSES/TOTAL
            TOTAL = arr[:4,:4].sum()
            print(files, FP, ACC, ACC2, MISCLASS_RATE)
            #Get confidence form file results_<conf>_<iou>
            #conf_thr == s
            fileName, _ = os.path.splitext(files)
            conf_thr = fileName.split('_')[1]
            iou_thr = fileName.split('_')[2]
            print(conf_thr, iou_thr)
            indx = int(float(conf_thr)*10)
            indy = int(float(iou_thr)*10)
            print(indx, indy)
            acc_results[indx, indy] = ACC
            exp_results[indx, indy] = ACC2
            miss_results[indx, indy] = MISCLASS_RATE
            coeff_results[indx, indy] = ACC/MISCLASS_RATE

    print(acc_results)
    print (exp_results)
    print(miss_results)


    my_plot(samples, acc_results, os.path.join(opt.folder,"{}.jpg".format("test_acc")), "accuracy")
    my_plot(samples, exp_results, os.path.join(opt.folder,"{}.jpg".format("test_exp")), "accuracy plus missclass")
    my_plot(samples, miss_results, os.path.join(opt.folder,"{}.jpg".format("test_miss")), "misclassifications")
    my_plot(samples, coeff_results, os.path.join(opt.folder,"{}.jpg".format("test_coeff")), "coefficient")
    my_plot2(acc_results, miss_results, os.path.join(opt.folder,"{}.jpg".format("test_state")), "accuracy", "misclassifications")

    results = np.argwhere(acc_results==acc_results.max())
    print(acc_results)

    with open(os.path.join(opt.folder,"rawresults.txt"), "a") as f:
        for r in results:
            print("MAX ACC: Confidence {} IOU {} ".format(0.1*r[0], 0.1*r[1]))
            print(acc_results[r[0], r[1]])
            f.write("MAX ACC: Confidence {} IOU {} \n".format(0.1*r[0], 0.1*r[1]))

    #plt.show()

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
