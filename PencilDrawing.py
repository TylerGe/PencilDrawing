import cv2
import numpy as np
import imutils # to get rotation of kernel
from scipy.sparse import spdiags as spdiags
from scipy.sparse.linalg import bicg as bicg
from PIL import Image

def GenStroke(im, ks, dirnum, gammaS):


    #sobelx = cv2.Sobel(im, cv2.CV_64F, 1, 0, ksize=3) # using sobel to get gradient is not a good idea maybe
    #sobely = cv2.Sobel(im, cv2.CV_64F, 0, 1, ksize=3)

    h, w = im.shape
    imx = abs(im[:,0:w-1] - im[:,1:w])
    imy = abs(im[0:h-1,:] - im[1:h,:])
    c = np.zeros((h,1))
    imx = np.column_stack((imx, c)) # add a column
    r = np.zeros((1,w))
    imy = np.row_stack((imy, r)) # add a row
    imedge = np.sqrt(imx*imx + imy*imy) # compute gradient

    kerref = np.zeros((ks*2+1,ks*2+1))
    kerref[ks,:] = 1
    ker = np.zeros((ks*2+1,ks*2+1,8))

    response = np.zeros((h,w,dirnum))
    for n in range(dirnum):
        ker[:,:,n] = imutils.rotate(kerref, n*180/dirnum)
        response[:,:,n] = cv2.filter2D(im,-1,ker[:,:,n]) # same as conv2, same size output

    index = response.argmax(axis=2) # store index of maximum response
    C = np.zeros((h,w,dirnum))
    for n in range(dirnum):
        C[:,:,n] = imedge*(1*(index==n)) # store the maximum response, maybe better approach

    spn = np.zeros((h,w,dirnum))
    for n in range(dirnum):
        spn[:,:,n] = cv2.filter2D(C[:,:,n],-1,ker[:,:,n])

    sp = spn.sum(axis=2) # sum 8 directions
    sp = (sp-np.min(sp))/(np.max(sp)-np.min(sp))
    s = 1-sp # invert and map to [0,1]
    return s**gammaS


Lambda = 0.2
texture_resize_ratio = 0.2

def GenTonemap(Y):
# parameters
	Ub = 225
	Ua = 105
    
	Mud = 90
    
	DeltaB = 9
	DeltaD = 11
    
	Omega1 = 42
	Omega2 = 29
	Omega3 = 29

	Omega1 = 52
	Omega2 = 37
	Omega3 = 11

	Omega1 = 76
	Omega2 = 22
	Omega3 = 2

	histgramTarget = np.zeros((256, 1),dtype=np.float32)
	histo = np.zeros((1, 256))
	ho=np.zeros((1, 256))
	hist_cv=np.zeros((1, 256))

	total = 0
#Compute target histgram
	for i in range(256):
		if i<Ua or i>Ub:
			p=0
		else:
			p=1/(Ub-Ua)
		histgramTarget[i,0]=(Omega1 * 1/DeltaB * np.e**(-(256-i)/DeltaB) +
			Omega2 * p +
			Omega3 * 1/np.sqrt(2 * np.pi * DeltaD) * np.e**(-(i-Mud)**2/(2*DeltaD**2))) * 0.01
		total=total+histgramTarget[i,0]
	histgramTarget[:,0]=histgramTarget[:,0]/total
	histgramTarget=np.transpose(histgramTarget)

	histo[0] = histgramTarget[0]
	# print(histgramTarget.shape)
	for i in range(1, 256):
		histo[0, i] = histo[0, i-1] + histgramTarget[0, i]

#imgae hist
	for i in range(256):
		hist_cv[0, i] = sum(sum(1 * (Y == i)))
	hist_cv /= float(sum(sum(hist_cv)))
	ho[0] = hist_cv[0]
	# print(hist_cv.shape)
	for i in range(1, 256):
		ho[0, i] = ho[0, i-1] + hist_cv[0, i]


	Iadjusted = np.zeros((Y.shape[0], Y.shape[1]))
	for x in range(Y.shape[0]):
		for y in range((Y.shape[1])):
			histogram_value = ho[0, Y[x, y]]
			index = (abs(histo - histogram_value)).argmin()
			Iadjusted[x, y] = index
	Iadjusted /= float(255)
	Iadjusted=cv2.blur(Iadjusted,(10,10))
	return Iadjusted

def GenPencil(im, P, J):
	h, w = im.shape
	P=cv2.resize(P,(w,h))
	bP=P
	P=np.reshape(P,(h*w,1))
	logP=np.log(P)
	logP=logP.T
	logP=spdiags(logP,0,h*w,h*w)

	J=cv2.resize(J,(w,h))
	J=J.reshape(h*w,1)
	logJ=np.log(J)
	e=np.ones((1,h*w))
	ddata=np.vstack((-e,e))
	Dx=spdiags(ddata,[0,h],h*w,h*w)
	Dy=spdiags(ddata,[0,1],h*w,h*w)
	A = 0.2*(Dx.dot(Dx.T)+Dy.dot(Dy.T))+np.dot(logP.T, logP)
	b=logP.T.dot(logJ)
	beta = bicg(A,b,tol=1) # tol parameter
	beta=beta[0].reshape(h,w)
	T=bP**beta
	T = (T - T.min()) / (T.max() - T.min())
	return T



def imgdemo(img):

	img = cv2.medianBlur(img, 3)

	Y=0.098*img[:,:,0]+0.504*img[:,:,1]+0.257*img[:,:,2]+16 # bgr order
	Cb=-0.148*img[:,:,2]-0.291*img[:,:,1]+0.439*img[:,:,0]+128
	Cr=0.439*img[:,:,2]-0.368*img[:,:,1]-0.071*img[:,:,0]+128

	S = GenStroke(Y, 2, 8, 2)
	Y = Y.astype(np.uint8)
	J = GenTonemap(Y) ** 1.2
	P = cv2.imread("texture.jpg",0)
	T = GenPencil(Y, P, J)
	R = S * T * 255
	cv2.imshow('1',np.uint8(R))

	# Recover color
	newY = R
	newim = img.copy()
	newim[:,:,0] = newY
	newim[:,:,1] = Cb
	newim[:,:,2] = Cr
	colorimg = cv2.cvtColor(newim, cv2.COLOR_YCR_CB2BGR)
	img = Image.fromarray(colorimg)
	img.show()
	cv2.waitKey (0)
	cv2.destroyAllWindows()

def videodemo():
	cap = cv2.VideoCapture('campus.mp4')
	out = cv2.VideoWriter('out.avi', -1, 20.0, (568,320))
	while (cap.isOpened()):
		ret, frame = cap.read()
		im = imgdemo(frame)
		out.write(im)
		cv2.imshow('frame', im)
		if cv2.waitKey(2) & 0xFF == ord('q'):
			break
	cap.release()
	out.release()


if __name__ == "__main__":
	img = cv2.imread('lena.jpg')
	imgdemo(img)
	# videodemo()

