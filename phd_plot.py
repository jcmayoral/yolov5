import argparse
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.Figure()

def process(opt):
    samples = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    #linspace for confidence threshold
    for ind,s in enumerate(samples):
        precision = np.zeros(len(samples))
        recall = np.zeros(len(samples))

        #look for files and read if matches with pattern
        for files in glob.glob(os.path.join(opt.folder,"results_"+str(s)+"*.txt")):
            #with open(files, "r") as f:
            #    text = f.readlines()
            #    print(text)
            print(files,s)
            #read txt file
            frame = pd.read_csv(files,delim_whitespace=True)
            fileName, _ = os.path.splitext(files)

            #Drop unused columns
            frame = frame.drop(['Images', 'Labels'], axis=1)

            #Get confidence form file results_<conf>_<iou>
            #conf_thr == s
            conf_thr = fileName.split('_')[1]
            iou_thr = fileName.split('_')[2]

            #index from iou_thr
            n = int(float(iou_thr)*10)
            print("INDEX", n)

            #iterate all, Lethal, Danger, Warning, Safe
            frame
            for cl in ['all']: #frame['Class'].unique():
                data = frame[frame['Class'] == cl]
                precision[n] = data['P']
                recall[n] = data['R']
                #print(data['P'], data['R'], data['mAP@.5'], data['mAP@.5:.95'])
        #sampling is the same in conf and iou
        print(precision, samples)
        plt.plot(samples, precision)
        plt.plot(samples, recall)
        plt.title("conf {} vs iou".format(s))
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
