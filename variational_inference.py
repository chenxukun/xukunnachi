import cv2
import numpy as np
from scipy.stats import norm
from math import log, exp

def read_data(filename, is_RGB, visualize=False, save=False, save_name=None):
	with open(filename, "r") as f:
		lines = f.readlines()

	data = []

	for line in lines:
		data.append(list(map(float, line.split(" "))))

	data = np.asarray(data).astype(np.float32)

	N, D = data.shape

	cols = int(data[-1, 0] + 1)
	rows = int(data[-1, 1] + 1)
	channels = D - 2
	img_data = data[:, 2:]

	# In numpy, transforming 1d array to 2d is in row-major order, which is different from the way image data is organized.
	image = np.reshape(img_data, [cols, rows, channels]).transpose((1, 0, 2))

	if visualize:
		if channels == 1:
			# for visualizing grayscale image
			cv2.imshow("", image)
		else:
			# for visualizing RGB image
			cv2.imshow("", cv2.cvtColor(image, cv2.COLOR_Lab2BGR))
		cv2.waitKey(0)
		cv2.destroyAllWindows()

	if save:
		if save_name is None:
			save_name = filename[:-4] + ".jpg"
		assert save_name.endswith(".jpg") or save_name.endswith(".png"), "Please specify the file type in suffix in 'save_name'!"

		if channels == 1:
			# for saving grayscale image
			cv2.imwrite(save_name, image)
		else:
			# for saving RGB image
			cv2.imwrite(save_name, (cv2.cvtColor(image, cv2.COLOR_Lab2BGR) * 255).astype(np.uint8))

	if not is_RGB:
		image = cv2.cvtColor(image, cv2.COLOR_Lab2RGB)

	return data, image
	
def write_img(image,filename,channels=1):
	save_name = filename + ".jpg"
	assert save_name.endswith(".jpg") or save_name.endswith(".png"), "Please specify the file type in suffix in 'save_name'!"

	if channels == 1:
		# for saving grayscale image
		cv2.imwrite(save_name, image)
	else:
		# for saving RGB image
		cv2.imwrite(save_name, (cv2.cvtColor(image, cv2.COLOR_Lab2BGR) * 255).astype(np.uint8))
		
def VIDenoise(img,name,n=5):
    vi = VI(img)
    vi.viprocess(name,n=n)
		
class VI:
    def __init__(self,img):
        self.width = img.shape[0]
        self.height = img.shape[1]
        self.img = img

        self.mus = np.ones((img.shape[0],img.shape[1]))*0.5#bernouli
        self.J = 2 #coupling strength
        self.vari = 1
        
    def neighbours(self,x,y):
        n=[]
        if (not x==0):
            n.append((x-1,y))
        if (not x==self.width-1):
            n.append((x+1,y))
        if (not y==0):
            n.append((x,y-1))
        if (not y==self.height-1):
            n.append((x,y+1))
            
        return n

    def pairwisePotential(self,p1,p2):
        (x1,y1) = p1
        (x2,y2) = p2
        xs = self.img[x1][y1]
        xt = self.img[x2][y2]
        #xs = 1*(xs>125)+(-1)*(xs<125)
        #xt = 1*(xt>125)+(-1)*(xt<125)

        return (self.J*xs*xt,self.J*(-xs)*xt)
    
    def viprocess(self,name,n=1):
        
        for i in range(n):
            self.img[self.img==255]=1
            self.img[self.img==0]=-1

            for x in range(self.img.shape[0]):
                for y in range(self.img.shape[1]):

                    xs = self.img[x][y]
                    #xs = 1*(xs>125)+(-1)*(xs<125)
                    
                    obv = self.img[x][y]
                    #obv = 1*(obv>125)+(-1)*(obv<125)
                    
                    pairwise = [self.pairwisePotential((x,y),p) for p in self.neighbours(x,y)]
                    firstPairwise = exp(sum([x[0] for x in pairwise]))
                    secondPairwise = exp(sum([x[1] for x in pairwise]))
                    pairwiseSum = firstPairwise+secondPairwise
                    firstPairwise/=pairwiseSum
                    secondPairwise/=pairwiseSum
                    #print('pair ',firstPairwise,secondPairwise)
                    
                    pxi = firstPairwise
                    pyilxi=norm(xs, self.vari).pdf(obv)
                    pyixi=pxi*pyilxi
                    
                    pxineg = secondPairwise
                    pyilxineg=norm((-xs), self.vari).pdf(obv)
                    pyixineg=pxineg*pyilxineg
                    up = pyixi
                    down = pyixineg
 
                    if(xs == 1):
                        up=pyixineg
                        down=pyixi
                    
                    ratio = up/down
                    newmu = 1/(ratio+1)
                    
                    self.mus[x][y] = newmu

                    
            for x in range(self.img.shape[0]):
                for y in range(self.img.shape[1]):
                    if (np.random.random()>self.mus[x][y]):
                        self.img[x][y] = -1
                    else:
                        self.img[x][y] = 1 

            print(i)
            self.img[self.img==1]=255
            self.img[self.img==-1]=0
            write_img(self.img,name+str(i))
	
if __name__=='__main__':
	data,img = read_data('a1/1_noise.txt',True,save=True)
	VIDenoise(img, 'vi_noise1_',n=5)
	write_img(img,'1_result_vi')

	data,img = read_data('a1/2_noise.txt',True,save=True)
	VIDenoise(img, 'vi_noise2_',n=5)
	write_img(img,'2_result_vi')

	data,img = read_data('a1/3_noise.txt',True,save=True)
	VIDenoise(img, 'vi_noise3_',n=5)
	write_img(img,'3_result_vi')

	data,img = read_data('a1/4_noise.txt',True,save=True)
	VIDenoise(img, 'vi_noise4_',n=5)
	write_img(img,'4_result_vi')

