# IFT6266 Project : inpainting

Blog link : https://stephanielarocque.github.io/ift6266_project/


## April 22th : Adding captions

Even though my L2 reconstruction network (similar to AE) is not as good at it can be (very blurry, maybe some other hyperparameters would be better), I wanted to know how much the captions could help. I used Francis Dutil's preprocessing to process the captions (remove stop words, only keep words occuring at least 10 times, switch to numbers) to get a vocabulary of +-7500 words. Next step to be able to use those processed captions was to find an embedding useful for this task.

The embedding I first tried is a bag-of-word.

Let's say we have the caption cap = [3, 45, 23, 8] (3rd word of the vocabulary, 45th, 23th and 8th word), it will be converted to a one hot bag of word : a vector 7500 zeros (size of vocabulary), except at position 3, 45, 23 and 8, where it will be a 1 (indicating that these words are present in the sentence). 

### 1 caption
As a starting point, I only used 1 caption per image. Each caption, as explained above, was a vector of +- 7500 digits. To obtain a useful representation, I used a dense layer from this vector to a vector of size 100. I then concatened this 100-vector to the 500-vector of latent variables obtained by convolution and pooling from the image's contour. With that new "latent code" of size 600, I used the same architecture as presented on April 15th post.

### 5 captions
Once I got this running, I thought it could be useful to use all the 5 captions instead of only 1. To do so, I extracted the 5 one hot bag of words for all of these 5 captions, so I had 5 vectors of size +- 7500, and then averaging them to get only 1 vector of size +-7500. I used that averaged bag of words as the captions' latent variables.

### Results
The results looks lot like without captions though.


** insert no cap
** insert 1 cap
** insert 5 cap

###

bla bla bla
==========


# April 15th : Basic Results

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

These are, unsuprisingly, blurry. Furthermore, the colors do not always match with the border (maybe because of batch norm with mode collape?). Using a GAN setup could help, even more if the whole images (border+center) are given to the discriminator : the colors would maybe match more the border. Using a GAN (adversarial loss) can also help to make the generated images sharper.




### Next steps
- Add the captions in this model and see if it helps 
- Use an adversarial loss (and L2 loss) as in this paper : https://arxiv.org/pdf/1604.07379.pdf



## March 20th : Gan abstract inpainting

My GAN model doesn't perform well. Since I thought my Jupyter Notebook was using GPU, but wasn't, I had problems because it just freezed my screen and everything when I tried to run 1 epoch. I finally got my GAN to "train", but NaNs occur after 2 or 3 epochs - and therefor *very* bad results for the generator.

![Gan crazy results](https://github.com/StephanieLarocque/ift6266_project/blob/master/blog_img_and_results/gan_crazy_result.png)



#### To do :
Maybe directly train using a DCGAN or Plug-and-Play network.

##  March 15th : First try implementing GAN

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



