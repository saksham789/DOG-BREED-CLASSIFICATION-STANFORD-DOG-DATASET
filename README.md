# DOG-BREED-CLASSIFICATION- STANFORD-DOG-DATASET
![alt image](https://user-images.githubusercontent.com/26468713/34918645-fbfe6bc2-f97b-11e7-9f89-548b508db905.jpg)
## DATASET
[Stanford Dog Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/) has around ~20 k images belonging to 120 classes and each image has an annotation associated with it.First Thought ,No. of images per breed availiable for training data which is roughly ~180 images, which is very less by the account of the Data required to train a Convolution Neural Net(CNN) classifier.
Since amount of Data we have is a constraint we use the concept of **Transfer Learning** ,which being said refers to technique which allows you to use the pretrained models on your own Dataset. Here we are going to use VGG16,VGG16BN(VGG16 with Batch Normalisation) models . VGG16 is a Deep CNN trained over Imagenet Dataset which has around 1000 synsets .
As, very well described in the paper [Visualizing and Understanding Convolutional Networks](https://arxiv.org/abs/1311.2901) the Bottom Layers of a Convolutional Neural Net activates only with primitive features (colour,texture,shape…) so these features can be transferred to other applications as well .Here we replace the top Layers(Fully Connected Layers and the softmax layers) and freeze the rest Layers so that they are non trainiable.Also we will make use of **Synthetic Imge Generation**  to take into account the randomness inthe images.
## Setting up the dataset
Download the [Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/) and extract and crop the images with the help of annotaions provided . Now we split the Dataset into Training,Validation and Testing,This should be done carefully ensuring there is no class imbalance in various chunks.Dataset can be converted into **tfRecords** Format as this format allows faster input and output operations. I won’t be explainig that here,See this [page](https://www.tensorflow.org/programmers_guide/datasets) for further references.
## Training
First we use a CNN Network and train it over the training data it with default parameters and Adam optimizer.
After 25 epochs:
**Training Accuracy:94.07%**
**Test Accuracy:51.07%**
Next we use Keras pretrained VGG16 model and replace the top Fully connected layers and a softmax Layer of 120 units since we have 120 classes.Make Sure you preprocess the input image the very same way it is done in the VGG16 paper. Now since the Bottom layers are frozen ,To avoid unnecessary computaion we can pass the input images once and save the Bootleneck features (The output of the Last convulational layer) .These Bottleneck features are then fed into the top model and the network is trained.
The Best Hyperparameters after tuning were :
**Learning_rate:1e-4**
After 50 epochs:
**Training Accuracy :97.8%**
**Test Accuracy:40.23%**
This clearly shows a presence of large variance or overfitting which can be mitigated by the use of Batch Normalization, L2 penalty or Dropout.Further we make use of Dropout And Batch Normalization with the help of the model VGG16BN .So, these regularizations methods gave us an 4% increase in the Test Accuracy but our model still seems to overfit the training data by huge amounts.Now it’s Deploy our Image Data Genertor . You can about this in deapth from [here](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html)
**Learning Rate:0.0001**
After 20 epochs
**Training Accyracy:88.23%**
**Test Accuracy:76.53%**
## PREDICTION USING YOLO
We trained our Data by cropping out the relevant part of the image using annotation file in Stanford dataset.Now while making prediction on a Random Image we can make useOf [Object Detection Algorithms like YOLO](https://pjreddie.com/darknet/yolo/) to locate the Bounding Box of a Dog in picture and then feed the cropped image to your model.
To Test the Accuracy Of Yolo ,we can make use of Annotaions in the Dataset images and Bounding Boxes  obtained by the YOLO algorithm.
**Accuracy Metric:INTERSECTION OVER UNION OF THE TWO BOXES**
## ERROR ANAlYSIS
Aah!! Let’s see where our model is getting wrong . We can do this by generating a Plotting a Confusion matrix like this:
![alt text](https://user-images.githubusercontent.com/26468713/34918643-fb86fa7e-f97b-11e7-88a7-7313bec9dd7d.png)

This doesn’t Look very clear so it’s very difficult to Visualise anything .Let’s see the top 30 misclassified pair of breeds.


![alt text](https://user-images.githubusercontent.com/26468713/34918644-fbc4d81c-f97b-11e7-98f8-dceac3203469.png)

As it can be seen, the pair ‘Silky Terrier / Yorkshire Terrier’ is the absolute leader in terms of misclassification which does make sense if we look into how these two breeds of dogs look like:


![alt text](https://user-images.githubusercontent.com/26468713/34918646-fc36fa14-f97b-11e7-9556-ecb2efda124e.png)

![alt text](https://user-images.githubusercontent.com/26468713/34918647-fc707096-f97b-11e7-961c-0ebddd2b6e4f.png)

This looks like the Optimal Bayes error rate which is Human in this case is itself very low because even humans confuse in this.For more details see this [article](https://www.cuteness.com/article/difference-between-yorkie-silky-terrier)
## Conclusion
We see how can achieve a train a Decent model with even modicum data with the Help of Transfer learning.
