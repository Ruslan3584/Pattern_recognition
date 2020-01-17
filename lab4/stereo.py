import sys
from PIL import Image
import numpy as np
import time

"""
alpha - коефіцієнт згладжування
maxd  - максимальний диспаритет
path_left,path_right  - ліва та права картинки
"""
alpha = int(sys.argv[1])
maxd = int(sys.argv[2])
path_left = sys.argv[3]
path_right = sys.argv[4]

# завантаження картинок 
print('Please,wait...')
left_img = np.array(Image.open(path_left).convert('L')).astype(float)
right_img = np.array(Image.open(path_right).convert('L')).astype(float)


# розміри картинок
shape_0 = left_img.shape[0]
shape_1 = left_img.shape[1]

# ваги вершин графу
def h(r,y,d):
    return abs(left_img[r,y] - right_img[r,y-d])

# ваги дуг
def g(d,d1):
    return abs(d - d1)

# алгоритм стеоро-зору
def recog(r): # r - номер стрічки
    e = np.empty(shape_1)     # диспаритет для однієї стрічки
    for m in range(1,shape_1):     # йдемо по всіх пікселях
        H[r,m,:] += (H[r,m-1,:] + G).min(axis=1) 

    e[0],e[-1] = np.argmin(H[r,-1,:]),0     # диспаритет 0-го пікселя завжди нуль, а останнього аргмін  
    for i in range(1,shape_1-1): # backward pass  відновлюємо диспаритети по штрафах H
        e[i]=np.argmin(H[r,-i,:]+G[int(e[i-1])]) # рахуємо диспаритети
    return e[::-1] # розвертаємо кожну стрічку,так як ідемо з кінця


G = np.empty((maxd+1,maxd+1))# матриця вагів дуг(пришвидшує роботу) 
for i in range(maxd+1):
    for j in range(maxd+1):
        G[i,j]= g(i,j)

G *= alpha # врахувуємо коефіцієнт згладжування 

a = time.time()

disparity = np.array(range(maxd+1)) # вектор можливих диспаритетів
H = np.array([h(i,j,disparity) for i in range(shape_0) for j in range(shape_1)]).reshape(shape_0,shape_1,maxd+1) # рахуємо всі можливі штрафи вершин 

time1 = time.time()-a
print('preprocessing time : ',time1,'sec')



a = time.time()
dispm = np.empty((shape_0,shape_1)) # diparity map
for j in range(shape_0): # диспаритети для кожної стрічки (стрічки незалежні)
    dispm[j] = recog(j)

time2 = time.time()-a
print('algorithm time : ',time2,'sec')
print('total time : ',time1 + time2,'sec')


# перетворюємо в картинку 
dm = np.array(dispm).reshape(left_img.shape[0],left_img.shape[1]).astype(np.uint8)

# нормалізація для правильного відображення (min = 0, max = 255 )
dm *=  int(255/dm.max())


# формуємо зображення в diparity_map.png
new_im = Image.fromarray(dm,mode = 'L')
new_im.save("diparity_map.png")

print('Result in diparity_map.png')
