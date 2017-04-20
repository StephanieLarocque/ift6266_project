# IFT6266 Project : inpainting

Blog link : https://stephanielarocque.github.io/ift6266_project/

## April 15th

Since my Vanilla GAN model is harder to train than expected, I started to code a basic convolutional net to obtain some results. 

My model is composed of 
- INPUT = Image border (64x64 with a black 32x32 square in the middle)
- Conv+Conv+Pooling
- Conv+Conv+Pooling
- Conv+Conv+Pooling
- Reshape Layer
- Dense Layer of size 500 (to obtain the latent variables)
- Dense Layer
- Reshape Layer
- Upscale+Conv+Conv
- Upscale+Conv+Conv
- OUTPUT = Image center (32x32)

Other specifications : all convolution layers use batch normalization
I used L2 loss as a starting point.


### Results
Here are the results for a subset of the validation set after 115 epochs :


![CNN basic results](https://github.com/StephanieLarocque/ift6266_project/blob/master/blog_img_and_results/cnn_basic_results.png)
Top = Input of the model, Middle = Ground truth, Bottom = Generated image

These are, unsuprisingly, blurry. 


### Next steps
- Add the captions in this model and see if it helps 
- Use an adversarial loss (and L2 loss) as in this paper : https://arxiv.org/pdf/1604.07379.pdf



## March 20th

My GAN model doesn't perform well. Since I thought my Jupyter Notebook was using GPU, but wasn't, I had problems because it just freezed my screen and everything when I tried to run 1 epoch. I finally got my GAN to "train", but NaNs occur after 2 or 3 epochs - and therefor *very* bad results for the generator.

![Gan crazy results](https://github.com/StephanieLarocque/ift6266_project/blob/master/blog_img_and_results/gan_crazy_result.png)



#### To do :
Maybe directly train using a DCGAN or Plug-and-Play network.

##  March 15th

These last weeks, I tried to implement a GAN (in order to extend it, once it works, to a conditional gan, for example)
#### Problem : 
Discriminator implementation in Lasagne 
#### Explanation : 
Since the discriminator takes the output of the generator AND true images, the "input" of the discriminator is not well defined in Lasagne 
#### Solution : 
We need to create 2 discriminators, once built on top of the generator, and the other one taking the true image as input that share weights (to simulate 1 single discriminator)
So I have built a GAN model that has 3 attributes:
  - G : Generator
  - D : Discriminator (input = true images), share weights with D_over_G
  - D_over_G : Discriminator (input = fake images = output from G), share weights with D



