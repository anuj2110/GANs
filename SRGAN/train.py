from networks import generator,discriminator
from keras.applications.vgg19 import VGG19
import matplotlib.pyplot as plt 
plt.switch_backend('agg')
from keras.models import Model 
import keras.backend as K 
from keras.layers import Input,Lambda 
from keras.optimizers import Adam,SGD,RMSprop
import cv2
import numpy as np 
import os

np.random.seed(10)
img_shape = (384,384,3)

def load_and_process_data(path,down_factor):
    imgs_hr=[]
    imgs_lr=[]
    for file in os.listdir(path):
        img = cv2.imread('/content/val2017/'+file)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img_hr = cv2.resize(img,(img_shape[0],img_shape[1]))
        img_lr = cv2.resize(img,(img_shape[0]//down_factor,img_shape[1]//down_factor))
        imgs_hr.append(img_hr)
        imgs_lr.append(img_lr)
    images_hr = np.array(imgs_hr[0:900])
    images_lr = np.array(imgs_lr[0:900])
    
    images_hr = (images_hr.astype(np.float32) - 127.5)/127.5
    images_lr = (images_lr.astype(np.float32) - 127.5)/127.5
    return (images_hr,images_lr)

def vgg_loss(y_true,y_pred):
    vgg19 = VGG19(include_top=False,weights='imagenet',input_shape=img_shape)
    vgg19.trainable=False
    for l in vgg19.layers:
        l.trainable=False
    loss_model = Model(inputs=vgg19.input,outputs=vgg19.get_layer('block5_conv4').output)
    loss_model.trainable = False
    return K.mean(K.square(loss_model(y_true) - loss_model(y_pred)))

def get_GAN(shape,discriminator,generator,optimizer):
    discriminator.trainable = False
    inp = Input(shape=shape)
    x = generator(inp)
    out = discriminator(x)
    GAN = Model(inputs=inp,outputs=[x,out])
    GAN.compile(loss=[vgg_loss,'binary_crossentropy'],loss_weights=[1,1e-3],optimizer=optimizer)
    return GAN
def denormalize(input_data):
    input_data = (input_data + 1) * 127.5
    return input_data.astype(np.uint8) 

path='/content/val2017'
hr,lr = load_and_process_data(path,4)

x_train_hr = hr[0:500]
x_train_lr = lr[0:500]

x_test_hr = hr[600:900]
x_test_lr = lr[600:900]

def plot_generated_images(epoch,generator, examples=3 , dim=(1, 3), figsize=(15, 5)):
    
    
    rand_nums = np.random.randint(0, x_test_hr.shape[0], size=examples)
    image_batch_hr = denormalize(x_test_hr[rand_nums])
    image_batch_lr = x_test_lr[rand_nums]
    gen_img = generator.predict(image_batch_lr)
    generated_image = denormalize(gen_img)
    image_batch_lr = denormalize(image_batch_lr)
    
    #generated_image = deprocess_HR(generator.predict(image_batch_lr))
    
    plt.figure(figsize=figsize)
    
    plt.subplot(dim[0], dim[1], 1)
    plt.imshow(image_batch_lr[1], interpolation='nearest')
    plt.axis('off')
        
    plt.subplot(dim[0], dim[1], 2)
    plt.imshow(generated_image[1], interpolation='nearest')
    plt.axis('off')
    
    plt.subplot(dim[0], dim[1], 3)
    plt.imshow(image_batch_hr[1], interpolation='nearest')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('output/gan_generated_image_epoch_%d.png' % epoch)

def train(epochs=1,batch_size=128):

if(not os.path.isdir('/content/output/')):
    os.mkdir('/content/output/')
downscale_factor = 4
batch_count = int(x_train_hr.shape[0] / batch_size)
shape = (img_shape[0]//downscale_factor, img_shape[1]//downscale_factor, img_shape[2])

gen = generator(shape).generator_model()
disc = discriminator(img_shape).discriminator_model()

adam = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
gen.compile(loss=vgg_loss, optimizer=adam)
disc.compile(loss="binary_crossentropy", optimizer=adam)

shape = (img_shape[0]//downscale_factor, img_shape[1]//downscale_factor, 3)
gan = get_GAN(shape, disc, gen, adam)

for e in range(1, epochs+1):
    print ('-'*15, 'Epoch %d' % e, '-'*15)
    for _ in range(batch_count):
        
        rand_nums = np.random.randint(0, x_train_hr.shape[0], size=batch_size)
        
        image_batch_hr = x_train_hr[rand_nums]
        image_batch_lr = x_train_lr[rand_nums]
        generated_images_sr = gen.predict(image_batch_lr)

        real_data_Y = np.ones(batch_size) - np.random.random_sample(batch_size)*0.2
        fake_data_Y = np.random.random_sample(batch_size)*0.2
        
        disc.trainable = True
        
        d_loss_real = disc.train_on_batch(image_batch_hr, real_data_Y)
        d_loss_fake = disc.train_on_batch(generated_images_sr, fake_data_Y)
        #d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)
        
        rand_nums = np.random.randint(0, x_train_hr.shape[0], size=batch_size)
        image_batch_hr = x_train_hr[rand_nums]
        image_batch_lr = x_train_lr[rand_nums]

        gan_Y = np.ones(batch_size) - np.random.random_sample(batch_size)*0.2
        disc.trainable = False
        loss_gan = gan.train_on_batch(image_batch_lr, [image_batch_hr,gan_Y])
        
    print("Loss HR , Loss LR, Loss GAN")
    print(d_loss_real, d_loss_fake, loss_gan)

    if e == 1 or e % 5 == 0:
        plot_generated_images(e, gen)
    if e % 50 == 0:
        generator.save('./output/gen_model%d.h5' % e)
        discriminator.save('./output/dis_model%d.h5' % e)
        gan.save('./output/gan_model%d.h5' % e)
train(20000,4)