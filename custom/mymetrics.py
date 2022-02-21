import torch
import numpy as np


def risk_loss(prob, pred, target):
    ### from objectiveness
    weights = torch.Tensor([[0., 1.0, 1.0, 1.0],
                            [0.5, 0., 0.75, 1.0],
                            [0.25, 0.5, 0., 0.75],
                            [0.25, 0.25, 0.5, 0.],
                            ]).view(-1, 4)
    results = torch.zeros(pred.size(0))
    ind = 0
    pred_prob = torch.sigmoid(pred)  # prob from logits

    for p,i,j in zip(pred_prob,pred, target):
        results[ind] = (p) * weights[i.int(),j.int()] #* torch.abs(i-j)
        ind +=1

    loss = torch.sum(results)/pred.size(0)
    #print("RISK LOSS{}".format(loss))
    #mult = torch.matmul(tensor1, tensor2)
    #print(mult)
    return loss

def risk_metrics(tp, conf, pred, target):
    # Sort by objectness
    #i = np.argsort(-conf)
    #Sort
    #tp, conf, pred_cls = tp[i], conf[i], pred[i]

    # Find unique classes
    unique_classes = np.unique(target) #get relevant classes
    print("shapes ", len(pred), len(target))

    print("shapes ", pred.shape(), target.shape())

    for t,p in zip(target, pred):
        print ("targets", t)
        print ("preditions ", p)

    #For in classes
    for ci, c in enumerate(unique_classes):
        #return all predictions of certain class
        i = pred == c
        print(i)
        print("class {}".format(c))
        n_l = (target == c).sum()  # number of labels from all targets
        n_p = i.sum()  # number of predictions
        print ("labels", n_l,  "detected predictions" , n_p)

        if n_p == 0 or n_l == 0:
            continue
        else:
            # Accumulate FPs and TPs
            #false positives true possitives arrays cumulative sum
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            print("tp", tpc)

            # Recall
            recall = tpc / (n_l + 1e-16)  # recall curve
            r[ci] = np.interp(-pr_score, -conf[i], recall[:, 0])  # r at pr_score, negative x, xp because xp decreases

            # Precision
            precision = tpc / (tpc + fpc)  # precision curve
            p[ci] = np.interp(-pr_score, -conf[i], precision[:, 0])  # p at pr_score

            # AP from recall-precision curve
            for j in range(tp.shape[1]):
                ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
                if j == 0:
                    py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5

    counts = torch.zeros(4)
    RISK_CLASS = torch.Tensor([[0, 1, 2, 3],
                            [1, 0, 1, 2],
                            [2, 1, 0, 2],
                            [3, 2, 1, 0],
                            ]).view(-1, 4)

    risks = torch.tensor((1,2,3,4))
    c_metric = torch.zeros(4)
    ind = 0

    for i,j in zip(pred, target):
        #print(i.byte(),j.long())
        #print(RISK_CLASS[i.long(),j.long()])
        counts[RISK_CLASS[i.long(),j.long()].long()]+=1

    return counts

if __name__=='__main__':
    nsamples = 100
    tensor1 = torch.randint(0,3,(nsamples,))
    tensor2 = torch.randint(0,4,(nsamples,))
    prob = torch.rand((nsamples,))
    risk_loss(prob,tensor1,tensor2)
    metrics = risk_metrics(tensor1,tensor2)
    for i,m in enumerate(metrics):
        print (i,m)
