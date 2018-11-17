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
		
def GibbsDenoise(img,name,burnin=4, loops=6):
    avg = np.zeros(img.shape)
    gb = Gibbs(img,avg)

    for i in range(img.shape[0]*img.shape[1]*loops):
        if(i%(img.shape[0]*img.shape[1]) ==0):
            print(i//(img.shape[0]*img.shape[1]))
        gb.gibbs_move(i,afterburnin=(i>=img.shape[0]*img.shape[1]*burnin))
        if(i%(img.shape[0]*img.shape[1])==0 and i>0):
            write_img(img,name+str(i/(img.shape[0]*img.shape[1])))
    
    for i in range(img.shape[0]*img.shape[1]):
        y = i//img.shape[0]
        x = i%img.shape[0]
        avg[x][y][0] = 255*(avg[x][y][0]>0)+0*(avg[x][y][0]<0)
    
    return avg

class Gibbs:
    def __init__(self,img,avg):
        self.width = img.shape[0]
        self.height = img.shape[1]
        self.img = img
        self.avg = avg
        self.J = 2
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
    
    def localEvidence(self,x,y):
        xs=1
        if(self.img[x][y]==0):
            xs = -1
            
        obv = xs
        return (norm(xs, self.vari).pdf(obv),norm(-xs, self.vari).pdf(obv))
        
        
    def pairwisePotential(self,p1,p2):
        (x1,y1) = p1
        (x2,y2) = p2
        xs = self.img[x1][y1]
        xt = self.img[x2][y2]
        if(xs == 255):
            xs=1
        else:
            xs=-1
        if(xt == 255):
            xt=1
        else:
            xt=-1
        return (np.exp(self.J*xs*xt),np.exp(self.J*(-xs)*xt))
        
    
    def gibbs_move(self,n, afterburnin=False):
        n = n%(self.width*self.height)
        y = n//self.width
        x = n%self.width
        localEvidence,negLocalEvidence = self.localEvidence(x,y)

        pairwise = [self.pairwisePotential((x,y),p) for p in self.neighbours(x,y)]
        firstPairwise = exp(sum(map(log,[x[0] for x in pairwise])))
        secondPairwise = exp(sum(map(log,[x[1] for x in pairwise])))
        
        firstPartP = localEvidence*firstPairwise
        secondPartP = negLocalEvidence*secondPairwise
        
        prob = firstPartP / (firstPartP+secondPartP)
        
        xs = self.img[x][y]
        if(xs == 0):
            prob = secondPartP / (firstPartP+secondPartP)

        if (np.random.random()>prob):
            self.img[x][y][0] = 0
            if(afterburnin):
                self.avg[x][y][0] += -1
        else:
            self.img[x][y][0] = 255
            if(afterburnin):
                self.avg[x][y][0] += 1

            
		
if __name__=='__main__':
	data,img = read_data('a1/1_noise.txt',True,save=True)
	denoiseImg = GibbsDenoise(img,'gib_noise1_',loops = 8)
	print('done')
	write_img(denoiseImg,'1_result_gib')

	data,img = read_data('a1/2_noise.txt',True,save=True)
	denoiseImg = GibbsDenoise(img,'gib_noise2_',loops = 8)
	print('done')
	write_img(denoiseImg,'2_result_gib')

	data,img = read_data('a1/3_noise.txt',True,save=True)
	denoiseImg = GibbsDenoise(img,'gib_noise3_',loops = 8)
	print('done')
	write_img(denoiseImg,'3_result_gib')

	data,img = read_data('a1/4_noise.txt',True,save=True)
	denoiseImg = GibbsDenoise(img,'gib_noise4_',loops = 8)
	print('done')
	write_img(denoiseImg,'4_result_gib')


