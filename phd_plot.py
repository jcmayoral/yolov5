import argparse
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class DataContainer:
    def __init__(self):
        self.initialize()

    def initialize(self):
        self.data = dict()
        for id in ["all", "Lethal", "Danger", "Warning", "Safe"]:
            self.data[id] = dict()
            for st in ['P', 'R','mAP@.5', 'mAP@.5:.95']:
                self.data[id][st] = np.zeros(11)

    def set(self, id, st, ind, value):
        self.data[id][st][ind] = value

    def get(self,id, st):
        return self.data[id][st]

def process(opt):
    samples = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    stats = ['P', 'R', 'mAP@.5', 'mAP@.5:.95']
    classes = ['all', 'Lethal', 'Danger', 'Warning', "Safe"]

    #linspace for confidence threshold
    for ind,s in enumerate(samples):
        container = DataContainer()
        #one for stats
        fig, axs = plt.subplots(2,2)
        fig.suptitle("Results Comparison NMS with Confidence Threshold set to  {}".format(s))

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

            #iterate all, Lethal, Danger, Warning, Safe
            frame
            for cl in frame['Class'].unique():
                data = frame[frame['Class'] == cl]
                for st in stats:
                    container.set(cl, st,n, data[st])
                #container.set(cl, 'R',n, data['R'])
                #container.set(cl, 'mAP@.5',n, data['mAP@.5'])
                #container.set(cl, 'mAP@.5:.95',n, data['mAP@.5:.95'])
                #print(data['P'], data['R'], data['mAP@.5'], data['mAP@.5:.95'])
        #sampling is the same in conf and iou
        #plt.plot(samples, precision)
        #plt.plot(samples, recall)
        for cl in classes:
            for i,st in enumerate(stats):
                r = i//2
                c = i%2
                axs[c,r].plot(samples, container.get(cl, st), label="Class {} Statics {}".format(cl,st))

        for i in range(4):
            r = i//2
            c = i%2
            axs[c,r].set_title(stats[i])
            axs[c,r].legend()
            axs[c,r].grid()
            axs[c,r].set_xlabel('IoU threshold')

        #plt.plot(samples, container.get('all', 'R'), label="All Recall")
        #plt.plot(samples, container.get('all', 'mAP@.5'), label="All mAP@.5")
        #plt.plot(samples, container.get('all', 'mAP@.5:.95'), label="All mAP@.5:95")

        #plt.title("conf {} vs iou".format(s))
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
