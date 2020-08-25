import caffe
import numpy as np

caffe.set_mode_cpu()

old_caffemodel="model/yolov3_mnasnet05_bigboom_0825.caffemodel"
old_prototxt_file="model/yolov3_mnasnet05_bigboom_0825.prototxt"


# new_caffemodel="model/yolov3_mnasnet05_bigboom_new.caffemodel"
# new_prototxt_file="model/yolov3_mnasnet05_bigboom.prototxt"


net = caffe.Net(old_prototxt_file, old_caffemodel, caffe.TEST)
# netNew = caffe.Net(new_prototxt_file, new_caffemodel, caffe.TEST)
netNew = net
for k, v in net.params.items():
    #print (k, v[0].data.shape)
    #print np.size(net.params[k])
    for i in range(np.size(net.params[k])):
        netNew.params[k][i].data[:] = np.copy(net.params[k][i].data[:])
        
params = {('610','610')}
for k, v in params:
    for i in range(np.size(net.params[v])):
        netNew.params[k][i].data[:] = 1 #np.copy(net.params[v][i].data[:])

params = {('635','635')}
for k, v in params:
    for i in range(np.size(net.params[v])):
        netNew.params[k][i].data[:] = 1 #np.copy(net.params[v][i].data[:])

netNew.save('model/yolov3_mnasnet05_bigboom_0825_new.caffemodel')