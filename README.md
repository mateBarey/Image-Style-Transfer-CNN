# Image-Style-Transfer-CNN

Recreating a Style Transfer using pytorch.

# Overview

In the notebook there are 2 photos a photo which contains the input content, and a photo of that contains the art style we 

would like to use in our image. Using a Convolutional Neural Network such as Vgg16, with CNN layers that have been trained to 

distinguish between a 1000 classes, we can extract the important features from each photo. 

The goal is to style the input reference image as the style reference imagesuch that the content in the image is conserved

but the style is changed.


The algorithm uses a CNN with 16 different CNN layers for extracting the generalized features of each image the , content image 

and the style image. The model is imported without its fully connected layers. The photo we want is generated based on 

solving an optimization problem which miminizes loss for 3 loss functions, style loss content loss, and total variation loss.

The content loss is a squared Euclidean distance between  the content image and a combined image of (content and style).


For the style loss we use something called a gram matrix . The matrix contains terms that are proportional to the 

covariances  of sets of features and emphasizes features that tend to activate together. By using the aggregated statistics 

across the image they're bound to the arrangement of the objects within the image. This is specifically what allows for the 

capture of information that emphasizes the style independent of the content of the image. 


The style loss is the Frobenius norm of the difference between gram matrices of style and the combination image

The total loss helps smooth out the noise of both the style and content loss.


The algorithm is very slow and can take up to 2 hours if only using a CPU and no GPU.
