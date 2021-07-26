# -*- coding: utf-8 -*-
"""
@author: yue.li
"""
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle

#parameters
n_input = 93
nzSDM_exp_preprocessing = np.load('nzSDM_exp_preprocessing_test.npy')
x_exp = nzSDM_exp_preprocessing.T
x_exp_dim = np.stack((x_exp[:, :n_input], x_exp[:, n_input:]), axis=2)
y_true = pd.read_csv('ROC_label.csv')
y = y_true.values
y_label = label_binarize(y, classes=[0, 1, 2])

#ROC analysis
n_classes = y_label.shape[1]
lw=2
colors = cycle(['aqua', 'darkorange', 'navy'])
class_names = cycle(['BCC', 'B2', 'D03'])
fig, ax = plt.subplots()
for i, color, class_name in zip(range(n_classes), colors, class_names):
    fpr = dict()
    tpr = dict()
    thresholds = dict()
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)    
   
    for k in range(5):
        model = tf.keras.models.load_model('saved_models_1CNN_layer_kernel_size_10_neuro_256_new_data_3class_multi_models/model_%s.h5'%(k)) 
        y_score = model.predict(x_exp_dim[:,:,:])
        fpr[i], tpr[i], thresholds[i] = roc_curve(y_label[:, i], y_score[:, i])
        roc_auc = auc(fpr[i], tpr[i])
        interp_tpr = np.interp(mean_fpr, fpr[i], tpr[i])
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(roc_auc)
        

    
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color=color,
            label= class_name+': mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=lw, alpha=.8)
    
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color=color, alpha=.2)
ax.plot([0, 1], [0, 1], 'k--', lw=lw, label='Chance', alpha=.8)    
ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
       title="Receiver operating characteristic to multi-class", xlabel='False Positive Rate', ylabel='True Positive Rate')
ax.legend(loc="lower right")       
plt.show()
