import numpy as np
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
def compute_iou(bbox1, bbox2):

    # TODO Compute IoU of 2 bboxes.
    bbox1=bbox1.detach().numpy()
    bbox2=bbox2.detach().numpy()
    s1=(bbox1[:,2]-bbox1[:,0])*(bbox1[:,3]-bbox1[:,1])
    s2=(bbox2[:,2]-bbox2[:,0])*(bbox2[:,3]-bbox2[:,1])
    a=np.maximum(bbox1[:,0],bbox2[:,0])
    b=np.maximum(bbox1[:,1],bbox2[:,1])
    c=np.minimum(bbox1[:,2],bbox2[:,2])
    d=np.minimum(bbox1[:,3],bbox2[:,3])
    op=np.maximum((c-a),0)*np.maximum((d-b),0)
    sm=s1+s2-op
    return op/sm
    ...

    # End of todo
