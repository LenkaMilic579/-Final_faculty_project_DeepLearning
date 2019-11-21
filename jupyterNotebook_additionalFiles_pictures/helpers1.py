# load all images in a directory and save as pixel values
from os import listdir
from PIL import Image
from numpy import asarray
# import cv2
import os
import io
from matplotlib import pyplot

def readAllPixels(num_of_dirs,dir_names):
    pixels_all = list()    
    m = 0
    for i in range(num_of_dirs):
        pixels_onefolder = list()
        for filename in listdir(dir_names[i]):  
            try:  
                image = Image.open(dir_names[i] + '/' + filename) 
                data = asarray(image) # convert image to numpy array
                pixels_onefolder.append(data)  
                m = m + 1
            except IOError: 
                print("error opening an image: %s",filename)
                pass   
        pixels_all.append(pixels_onefolder)
    return pixels_all, m

def remove_img(array_of_paths):
    a = True
    for p in array_of_paths:
        # check if file exists or not
        if os.path.exists(p) is False :
            a = False
        else:
            os.remove(p)
    print(a)

def FlipIOmages(folderi, imena_foldera):   
    c = 0
    for i in range(26):
        if(len(folderi[i]) < 100):
            path = str(imena_foldera[i])
            for elem in folderi[i]:
                c = c + 1
                image = cv2.imread(elem, -1)
                hoz_flip = cv2.flip(image, flipCode=1)
                string = 'hoz_'+str(c)+'.jpg'
                cv2.imwrite(os.path.join(path , string), hoz_flip)
                cv2.waitKey(0)

def CropImages_makeSquare(names_of_pics, imena_foldera):
    path = 'DONE'
    c = 0
    for elem in names_of_pics:
        c = c + 1
        string = 'w_'+str(c)
        image = Image.open(elem)
        sirina = image.size[0]
        visina = image.size[1]
        #       Ustanovi koja stranica je kraca, zatim drugu smanji na tu dimenziju
        if visina < sirina:
            box = (0, 0, visina, visina)
            image = Image.open(elem)
            cropped_image = image.crop(box)
            if(cropped_image.mode != "RGB"):
                im = cropped_image.convert("RGB")
                im.save(string+'.jpg')
            else:
                cropped_image.save(string+'.jpg')
        elif visina > sirina:
            box = (0, 0, sirina, sirina)
            image = Image.open(elem)
            cropped_image = image.crop(box)
            if(cropped_image.mode != "RGB"):
                im = cropped_image.convert("RGB")
                im.save(string+'.jpg')
            else:
                cropped_image.save(string+'.jpg')
        else:
            im.save(string+'.jpg')
            pass

def resize_aspect_fit(path,final_size):
    dirs = os.listdir( path )
    for item in dirs:
         if item == '.DS_Store':
             continue
         if os.path.isfile(path+item):
             im = Image.open(path+item)
             f, e = os.path.splitext(path+item)
             size = im.size
             ratio = float(final_size) / max(size)
             new_image_size = tuple([int(x*ratio) for x in size])
             im = im.resize(new_image_size, Image.ANTIALIAS)
             new_im = Image.new("RGB", (final_size, final_size))
             new_im.paste(im, ((final_size-new_image_size[0])//2, (final_size-new_image_size[1])//2))
             new_im.save(f + 'res200.jpg', 'JPEG', quality=90)

def solve_fc(p_all, layers_dims_conv):
    params_fc = {}
    L1 = len(layers_dims_conv) + 1
    L2 = len(p_all) // 2 + 1
    for l in range(L1, L2):
        params_fc['W' + str(l-2)] = p_all['W' + str(l)]
        params_fc['b' + str(l-2)] = p_all['b' + str(l)]
    return params_fc

def solve_conv(p_all, layers_dims_conv, channels, nc):   
    p_conv = {}
    L = len(layers_dims_conv) 
    for l in range(L):
        p_conv_tempw = {key:p_all[key] for key in ['W' + str(l+1)]}
        p_conv_tempb = {key:p_all[key] for key in ['b' + str(l+1)]}
        p_conv['W' + str(l+1)] = p_conv_tempw['W' + str(l+1)].reshape(layers_dims_conv[l], layers_dims_conv[l], channels[l], nc[l])
        p_conv['b' + str(l+1)] = p_conv_tempb['b' + str(l+1)].reshape(1, 1, 1, nc[l])             
    return p_conv

def solve_all(p_c, p_fc):
    parameters_all = {}    
    L1 = len(p_c) // 2
    L2 = len(p_fc) // 2 + 1
    for l in range(1, L1+1):
        w = p_c['W' + str(l)]
        b = p_c['b' + str(l)]
        parameters_all['W' + str(l)] = w.reshape(w.shape[3], -1)
        parameters_all['b' + str(l)] = b.reshape(b.shape[3], -1)
    for l in range(1, L2):
        w = p_fc['W' + str(l)]
        b = p_fc['b' + str(l)]
        parameters_all['W' + str(l+L1)] = w
        parameters_all['b' + str(l+L1)] = b
    return parameters_all
def solve_all_grads(g_fc,dA1,dW1,db1,dA2,dW2,db2):
    g_all = {"dA1": dA1.reshape(dA1.shape[3], -1),"dW1": dW1.reshape(dW1.shape[3], -1), "db1": db1.reshape(db1.shape[3], -1),
             "dA2": dA2.reshape(dA2.shape[3], -1),"dW2": dW2.reshape(dW2.shape[3], -1), "db2": db2.reshape(db2.shape[3], -1)}
    L2 = len(g_fc) // 3 
    for l in range(1, L2+1):
        g_all['dA' + str(l+2)] = g_fc['dA' + str(l)]
        g_all['dW' + str(l+2)] = g_fc['dW' + str(l)]
        g_all['db' + str(l+2)] = g_fc['db' + str(l)]
    return g_all