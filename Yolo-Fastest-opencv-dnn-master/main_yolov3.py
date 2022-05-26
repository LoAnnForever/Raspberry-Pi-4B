import cv2 as cv
import argparse
import numpy as np

cap = cv.VideoCapture(0)#调用摄像头
#cap = cv.VideoCapture('img00007.jpg')#读取视频或图片
# 创建可以调节大小的窗口
#fps = 7
fps = cap.get(cv.CAP_PROP_FPS)#获取视频流的FPS
# 创建可以调节大小的窗口
size = (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)))
#videoWriter = cv.VideoWriter('sq.avi', cv.VideoWriter_fourcc('X','V','I','D'), fps, size)
#cv.namedWindow("AIM", 0)
#cv.resizeWindow("AIM", 700, 700)
#cap.set(cv.CAP_PROP_FPS, 60)
# 初始化参数
confThreshold = 0.25  # Confidence threshold 信心阈值
nmsThreshold = 0.3  # Non-maximum suppression threshold 抑制阈值
inpWidth = 320  # Width of network's input image 网络输入图像的宽度
inpHeight = 320  # Height of network's input image 网络输入图像的高度

# Give the configuration and weight files for the model and load the network using them.
# 给出模型的配置和权重文件，并使用它们加载网络
#modelConfiguration = "Yolo-Fastest-voc/yolo-fastest-xl.cfg"
#modelWeights = "Yolo-Fastest-voc/yolo-fastest-xl.weights"

modelConfiguration = "Yolo-Fastest-coco/yolo-fastest-xl.cfg"
modelWeights = "Yolo-Fastest-coco/yolo-fastest-xl.weights"

# Load names of classes 加载类名
#classesFile = "voc.names"
classesFile = "coco.names"
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')
    colors = [np.random.randint(0, 255, size=3).tolist() for _ in range(len(classes))]
#print(classes)
#print(colors)
# Get the names of the output layers 获取输出层的名称
def getOutputsNames(net):
    # Get the names of all the layers in the network 获取网络中所有层的名称
    layersNames = net.getLayerNames()
    # print(dir(net))
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    # 获取输出层的名称，即输出不相连的层     
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Draw the predicted bounding box 画出预测的边界框
# def drawPred(frame,classId, conf, left, top, right, bottom):
def drawPred(classId, conf, left, top, right, bottom):
    # Draw a bounding box.绘制一个边框。
    
    label = '%.2f' % conf
#    log = ''
    if classes and classes[classId]=='person':
#    if (classes and classes[classId]=='person')or(classes and classes[classId]=='car'):

        assert (classId < len(classes))
        label = '%s:%s' % (classes[classId], label)
        cv.rectangle(frame, (left, top), (right, bottom),colors[classId], thickness=2)

    # Display the label at the top of the bounding box 将标签显示在边框顶部
        labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
    # cv.rectangle(frame, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]), top + baseLine), (255,255,255), cv.FILLED)
        cv.putText(frame, label, (left, top-10), cv.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255), thickness=2)
    else:
        pass
# Remove the bounding boxes with low confidence using non-maxima suppression
# 使用非极大值抑制删除低置信度的边界框
def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    classIds = []
    confidences = []
    boxes = []
    # Scan through all the bounding boxes output from the network and keep only the ones with high confidence scores.Assign the box's class label as the class with the highest score.
    # 扫描网络输出的所有包围框，只保留那些有高自信分数的框。将方框上的类标签指定为得分最高的类    
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with lower confidences.
    # 执行非最大抑制，以消除较低的置信度的冗余重叠框。    
    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        drawPred(classIds[i], confidences[i], left, top, left + width, top + height)
while(1):
    # Process inputs 过程的输入
    net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
    try:
        # 读取一帧,如果有剩余帧ret为ture,否则为false
        ret, frame = cap.read()
        blob = cv.dnn.blobFromImage(frame, 1 / 255.0, (inpWidth, inpHeight), [0, 0, 0], swapRB=False, crop=False)
    except:
        break
    net.setInput(blob)
    outs = net.forward(getOutputsNames(net))
    postprocess(frame, outs)
    t, _ = net.getPerfProfile()
    ti=t * 1000.0 / cv.getTickFrequency()
    label = ''
    label = 'FPS: %.2f' % (1000.0/ti)
    cv.putText(frame,label,(0, 25), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255),thickness=2)
    cv.imshow("video", frame)
#    videoWriter.write(frame)
    cv.imwrite('Asava_pic/object.jpg', frame)
    # waitKey default 1ms  or(AlarmStatus==False)
    if (cv.waitKey(1) & 0xFF == ord('q')):
        break
cap.release()
cv.destroyAllWindows()
  