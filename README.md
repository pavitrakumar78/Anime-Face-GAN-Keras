# Anime-Face-GAN-Keras

A DCGAN to generate anime faces using custom dataset in Keras.  

# Dataset  

The dataset is created by crawling anime database websites using `curl`. The script `anime_dataset_gen.py` crawls and processes the images into 64x64 PNG images with only the faces cropped.  

## Examples of the dataset:  
 ![trainimg1.png](https://github.com/pavitrakumar78/Anime-Face-GAN-Keras/blob/master/images/train_img1.png)
 ![trainimg2.png](https://github.com/pavitrakumar78/Anime-Face-GAN-Keras/blob/master/images/train_img2.png)
 ![trainimg3.png](https://github.com/pavitrakumar78/Anime-Face-GAN-Keras/blob/master/images/train_img3.png)
 
 # Network  

This implementation of GAN uses deconv layers in Keras (networks are initialized in the `GAN_Nets.py` file). I have tried various combinations of layers such as :  
Conv + Upsampling  
Conv + bilinear  
Conv + Subpixel Upscaling  
But none of these combinations yielded any decent results. The case was either GAN fails to generate images that resembles faces or it generates same or very similar looking faces for all batches (generator collapse). But these were my results, maybe techniques such as mini-batch discrimination, z-layers could be used to get better results.  

# Training

Only simple GAN training methods are used. Training is done on about 22,000 images. Images are not loaded entirely into memory instead, each time a batch is sampled, only the sampled images are loaded. An overview of what happens each step is:  
-Sample images from dataset (real data)  
-Generate images using generator (gaussian noise as input) (fake data)  
-Add noise to labels of real and fake data  
-Train discriminator on real data 
-Train discriminator on fake data  
-Train GAN on fake images and real data labels  
Training is done for 10,000 steps. In my setup (GTX 660; i5 4670) it takes 10-11 secs for each step.  

## Loss plot:

![realvsfakeloss.png](https://github.com/pavitrakumar78/Anime-Face-GAN-Keras/blob/master/images/realvsfakeloss.png)

![genloss.png](https://github.com/pavitrakumar78/Anime-Face-GAN-Keras/blob/master/images/genloss.png)

## Full Training as a GIF: (images sampled every 100 step)

![movie.gif](https://github.com/pavitrakumar78/Anime-Face-GAN-Keras/blob/master/images/movie.gif)

## Faces generated at the end of 10,000 steps:

![finalimg2.png](https://github.com/pavitrakumar78/Anime-Face-GAN-Keras/blob/master/images/final_img2.png)
![finalimg3.png](https://github.com/pavitrakumar78/Anime-Face-GAN-Keras/blob/master/images/final_img3.png)
![finalimg4.png](https://github.com/pavitrakumar78/Anime-Face-GAN-Keras/blob/master/images/final_img4.png)

The faces look pretty good IMO, might look more like an actual face with more training, more data and probably with a better network.

# Resources
https://github.com/tdrussell/IllustrationGAN  
https://github.com/jayleicn/animeGAN  
https://github.com/forcecore/Keras-GAN-Animeface-Character  
  
https://distill.pub/2016/deconv-checkerboard/  
https://kivantium.net/keras-bilinear  
