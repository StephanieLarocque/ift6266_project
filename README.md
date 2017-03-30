# IFT6266 Project : inpainting

Blog link : https://stephanielarocque.github.io/ift6266_project/



## 1st blog post -- March 15th

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

## 2nd blog post -- March 20th

My GAN model doesn't perform well. Since I thought my Jupyter Notebook was using GPU, but wasn't, I had problems because it just freezed my screen and everything when I tried to run 1 epoch. 

