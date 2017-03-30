# IFT6266 Project : inpainting

Blog link : https://stephanielarocque.github.io/ift6266_project/

These last weeks, I tried to implement a GAN (in order to extend it, once it works, to a conditional gan, for example)

## 1st blog post -- March 15th
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

## 2nd blog post -- date
