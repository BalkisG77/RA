import cv2
import numpy as np
import math
from objloader_simple import *
from pyzbar.pyzbar import decode

cap = cv2.VideoCapture(0)
imgTarget = cv2.imread('img/carte.jpeg')
imgTarget2 = cv2.imread('img/carte2.jpeg')
imgTarget3 = cv2.imread('img/carte3.jpeg')
myVid = cv2.VideoCapture('video/eau.mp4')
myImg = cv2.imread('img/lotus.jpg')

with open('myDataFile.txt') as f:
    myDataList = f.read().splitlines()

width  = cap.get(3)  # float `width`
height = cap.get(4)  # float `height`

camera_parameters = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])
DEFAULT_COLOR = (0, 0, 0)

detection = False
frameCounter = 0

#3D
obj = OBJ(('model/wolf.obj'), swapyz=True) 

#Video
success, imgVideo = myVid.read()
hT,wT,cT = imgTarget.shape
imgVideo = cv2.resize(imgVideo,(wT,hT))

#Photo 
hT2,wT2,cT2 = imgTarget2.shape
myImg = cv2.resize(myImg,(wT,hT))

#Photo2 
hT3,wT3,cT3 = imgTarget3.shape

orb = cv2.ORB_create(nfeatures=1000)
#Video
kp1, des1 = orb.detectAndCompute(imgTarget,None)
#Photo
kpImg, desImg = orb.detectAndCompute(imgTarget2,None)
#Model
kpModel, desModel = orb.detectAndCompute(imgTarget3,None)
#imgTarget = cv2.drawKeypoints(imgTarget,kp1,None)

def stackImages(imgArray,scale,lables=[]):
    sizeW= imgArray[0][0].shape[1]
    sizeH = imgArray[0][0].shape[0]
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                imgArray[x][y] = cv2.resize(imgArray[x][y], (sizeW,sizeH), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((sizeH, sizeW, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
            hor_con[x] = np.concatenate(imgArray[x])
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    else:
        for x in range(0, rows):
            imgArray[x] = cv2.resize(imgArray[x], (sizeW, sizeH), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        hor_con= np.concatenate(imgArray)
        ver = hor
    if len(lables) != 0:
        eachImgWidth= int(ver.shape[1] / cols)
        eachImgHeight = int(ver.shape[0] / rows)
        print(eachImgHeight)
        for d in range(0, rows):
            for c in range (0,cols):
                cv2.rectangle(ver,(c*eachImgWidth,eachImgHeight*d),(c*eachImgWidth+len(lables[d])*13+27,30+eachImgHeight*d),(255,255,255),cv2.FILLED)
                cv2.putText(ver,lables[d],(eachImgWidth*c+10,eachImgHeight*d+20),cv2.FONT_HERSHEY_COMPLEX,0.7,(255,0,255),2)
    return ver
def render(img, obj, projection, model, color=False):
    """
    Render a loaded obj model into the current video frame
    """
    vertices = obj.vertices
    scale_matrix = np.eye(3) * 3
    h, w, c = model.shape

    for face in obj.faces:
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        points = np.dot(points, scale_matrix)
        # render model in the middle of the reference surface. To do so,
        # model points must be displaced
        points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
        imgpts = np.int32(dst)
        if color is False:
            cv2.fillConvexPoly(img, imgpts, DEFAULT_COLOR)
        else:
            color = hex_to_rgb(face[-1])
            color = color[::-1]  # reverse
            cv2.fillConvexPoly(img, imgpts, color)

    return img

def projection_matrix(camera_parameters, homography):
    # Compute rotation along the x and y axis as well as the translation
    homography = homography * (-1)
    rot_and_transl = np.dot(np.linalg.inv(camera_parameters), homography)
    col_1 = rot_and_transl[:, 0]
    col_2 = rot_and_transl[:, 1]
    col_3 = rot_and_transl[:, 2]
    # normalise vectors
    l = math.sqrt(np.linalg.norm(col_1, 2) * np.linalg.norm(col_2, 2))
    rot_1 = col_1 / l
    rot_2 = col_2 / l
    translation = col_3 / l
    # compute the orthonormal basis
    c = rot_1 + rot_2
    p = np.cross(rot_1, rot_2)
    d = np.cross(c, p)
    rot_1 = np.dot(c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_2 = np.dot(c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_3 = np.cross(rot_1, rot_2)
    # finally, compute the 3D projection matrix from the model to the current frame
    projection = np.stack((rot_1, rot_2, rot_3, translation)).T
    return np.dot(camera_parameters, projection)

while True:
    success, imgWebcam = cap.read()
    imgAug = imgWebcam.copy()
    #Creation des points de reconnaissance
    kp2, des2 = orb.detectAndCompute(imgWebcam,None)
    #imgWebcam = cv2.drawKeypoints(imgWebcam,kp2,None)

    if detection == False:
        myVid.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frameCounter = 0
    else:
        if frameCounter == myVid.get(cv2.CAP_PROP_FRAME_COUNT):
            myVid.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frameCounter = 0
        success, imgVideo = myVid.read()
        imgVideo = cv2.resize(imgVideo,(wT,hT))


    #Code pour calculer le taux de match entre la cam et l'image cible
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append(m)

    matches2 = bf.knnMatch(desImg,des2,k=2)
    good2 = []
    for m,n in matches2:
        if m.distance < 0.75*n.distance:
            good2.append(m)

    matches3 = bf.knnMatch(desModel,des2,k=2)
    good3 = []
    for m,n in matches3:
        if m.distance < 0.75*n.distance:
            good3.append(m)

    imgFeatures = cv2.drawMatches(imgTarget,kp1,imgWebcam,kp2,good,None,flags=2)
    print(len(good))

    #VIDEO
    if len(good) > 20:
        detection = True
        srcPts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        dstPts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
        matrix, mask = cv2.findHomography(srcPts,dstPts,cv2.RANSAC,5)
        print(matrix)

        pts = np.float32([[0,0],[0,hT],[wT,hT],[wT,0]]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,matrix)
        #img2 = cv2.polylines(imgWebcam,[np.int32(dst)],True,(255,0,255),3)

        imgWarp = cv2.warpPerspective(imgVideo,matrix,(imgWebcam.shape[1],imgWebcam.shape[0]))
        
        maskNew = np.zeros((imgWebcam.shape[0],imgWebcam.shape[1]),np.uint8)
        cv2.fillPoly(maskNew,[np.int32(dst)],(255,255,255))
        maskInv = cv2.bitwise_not(maskNew)
        imgWebcam = cv2.bitwise_and(imgWebcam,imgWebcam,mask = maskInv)
        imgWebcam = cv2.bitwise_or(imgWarp, imgWebcam)

        imgStacked = stackImages(([imgWebcam, imgTarget],[imgFeatures, imgAug]),0.5)
        #cv2.imshow('Image',imgAug)
    else :
        detection = False

    #IMAGE
    if len(good2) > 20: 
        srcPts = np.float32([kpImg[m.queryIdx].pt for m in good2]).reshape(-1,1,2)
        dstPts = np.float32([kp2[m.trainIdx].pt for m in good2]).reshape(-1,1,2)
        matrix, mask = cv2.findHomography(srcPts,dstPts,cv2.RANSAC,5)
        print(matrix)

        pts = np.float32([[0,0],[0,hT2],[wT2,hT2],[wT2,0]]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,matrix)
        #img2 = cv2.polylines(imgWebcam,[np.int32(dst)],True,(255,0,255),3)

        imgWarp = cv2.warpPerspective(myImg,matrix,(imgWebcam.shape[1],imgWebcam.shape[0]))
        
        maskNew = np.zeros((imgWebcam.shape[0],imgWebcam.shape[1]),np.uint8)
        cv2.fillPoly(maskNew,[np.int32(dst)],(255,255,255))
        maskInv = cv2.bitwise_not(maskNew)
        imgWebcam = cv2.bitwise_and(imgWebcam,imgWebcam,mask = maskInv)
        imgWebcam = cv2.bitwise_or(imgWarp, imgWebcam)

        imgStacked = stackImages(([imgWebcam, imgTarget],[imgFeatures, imgAug]),0.5)
        #cv2.imshow('Image',imgWebcam)

    #MODEL
    if len(good3) > 20: 
        print("ok good3")
        srcPts = np.float32([kpModel[m.queryIdx].pt for m in good3]).reshape(-1,1,2)
        dstPts = np.float32([kp2[m.trainIdx].pt for m in good3]).reshape(-1,1,2)
        matrix, mask = cv2.findHomography(srcPts,dstPts,cv2.RANSAC,5)
        print(matrix)

        #pts = np.float32([[0,0],[0,hT3],[wT3,hT3],[wT3,0]]).reshape(-1,1,2)
        #dst = cv2.perspectiveTransform(pts,matrix)
        #img2 = cv2.polylines(imgWebcam,[np.int32(dst)],True,(255,0,255),3)

        projection = projection_matrix(camera_parameters, matrix)  

        frame = render(imgWebcam, obj, projection, imgTarget3, False)
                
        cv2.imshow('Image',frame)

    #QRCODE
    for barcode in decode(imgWebcam):
        myData = barcode.data.decode('utf-8')
        print(myData)

        if myData in myDataList:
            print('Authorized')
            myOutput = 'Authorized'
            myColor = (0, 255, 0)
        else:
            print('Un-Authorized')
            myOutput = 'Un-Authorized'
            myColor = (0, 0, 255)

        pts = np.array([barcode.polygon], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(imgWebcam, [pts], True, myColor, 5)
        pts2 = barcode.rect
        cv2.putText(imgWebcam, myOutput, (pts2[0], pts2[1]), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, myColor, 2)

    cv2.imshow('Result', imgFeatures)


    #cv2.imshow('Comparaison',imgWarp)
    #cv2.imshow('Mon image ref',imgTarget)
    #cv2.imshow('video', imgVideo)
    #cv2.imshow('cam', imgWebcam )
    cv2.waitKey(1)
    frameCounter += 1


