# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 13:41:52 2020

@author: gcass
"""

from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys
import os
import glob

def IOU (bb1, bb2):
    interW = min(bb1[0],bb2[0])
    interH = min(bb1[1], bb2[1])
    


    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = interW * interH
    union_area  = bb1[0]*bb1[1] + bb2[0]*bb2[1] - intersection_area
    
    iou = intersection_area / union_area
    print(iou)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou
    

def add_boxes_dims(boxFileImg, dimsRecord):
    with open(boxFileImg) as f :
        boxes = f.read().splitlines() 
        for box in boxes :
            dimsRecord.append(box.split(' ')[3:5])
            
def compute_anchors(boxDir, maxCluster):
    fullBoxDir    = os.path.join("boxes",boxDir)
    fullAnchorDir = os.path.join("anchors",boxDir)
    
    dimRecord = []
    
    for boxes in glob.glob(fullBoxDir+"/*.txt"):
        add_boxes_dims(boxes, dimRecord)
        
    """
    jusqu'ici les dimension des boxes sont stockées sous forme de chaine de caractère dans une liste de couple 
    de dimensionn (largeur, heuteur)
    """    
    dimRecord       = np.array(dimRecord).astype(float)
    
    
    
    plotDimRec = np.transpose(dimRecord)
    
    Hist2D = plt.figure(1)
    plt.hist2d(plotDimRec[0], plotDimRec[1], bins = [50,50])
    plt.show()
    
    pointsFigure = plt.figure(0)
    plt.scatter(plotDimRec[0], plotDimRec[1])
    plt.title("Repartition of box dimension")
    
    clustersInertia = {}
    colors =['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    """
    for potentialNumberAnchor in range (1,maxCluster+1): 
        KM = KMeans(potentialNumberAnchor).fit(dimRecord)
        inertia = KM.inertia_
        potentialAnchors  = KM.cluster_centers_
        plt.scatter(potentialAnchors[:,0], potentialAnchors[:,1], color = colors[potentialNumberAnchor])
        clustersInertia.update({potentialNumberAnchor : inertia})
    """
    potentialAnchors = DBSCAN(eps = 0.05,metric = IOU).fit(dimRecord).components_
    plt.scatter(potentialAnchors[:,0], potentialAnchors[:,1], color = colors[3])
    plt.show()
    
    
    """
    varianceFigure = plt.figure(2)
    print(list(clustersInertia.values()))
    plt.plot(list(clustersInertia.values()))
    plt.title("Variance")
    """
    
    derivate1_inertia = {i : clustersInertia[i-1]-clustersInertia[i] for i in range(2,maxCluster+1)}
    derivate2_inertia = {i : derivate1_inertia[i]/derivate1_inertia[i+1] for i in range(2,maxCluster)}
    bestNumberAnchors = max(derivate2_inertia, key = derivate2_inertia.get)
    
    sys.stdout.write("this is the best number of anchors : {}".format(bestNumberAnchors))  # same as print
    sys.stdout.flush()
    
    
    plt.show()
    
    anchors           = KMeans(bestNumberAnchors).fit(dimRecord).cluster_centers_
    #np.savetxt(fullAnchorDir, anchors, delimiter = ",", fmt = "%10.5f")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--boxDir", help ="directory of the ground truth boxes")
    parser.add_argument("--maxCluster", type = int)
    args = parser.parse_args()
    
    compute_anchors(args.boxDir, args.maxCluster)
    
    
if __name__ == "__main__":
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    
    compute_anchors("TRUE_TEST/data",8)
    #main()
    
     
    








