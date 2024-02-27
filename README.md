# Brain-tumor-image-classification

This project aims at finding the best deep learning model to classify brain MRI images according to if there is a tumor or not. The data set is made of 196 images of brain, and can be found [here](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection/data). 

## Data pre-processing
After renaming the images and randomly split them between training and testing, I resized the images so they were all the size of the smallest image of the data set (150x150).

## Modeling process

### Existing models
I fine-tuned 11 existing convolutional neural networks, focusing on three well-known architectures: EfficientNet, ResNet, and VGG. I trained different versions of each to compare their performance and complexity variations across architectures.

The existing models differ in architecture and complexity, leading to variations in performance depending on dataset characteristics.
- Among the three architectures tested, VGG is the shallowest, ranging from 11 to 19 layers. It utilizes small convolutional filters but has a large number of parameters (533MB for VGG16) due to a high number of fully connected layers. VGG is well-suited for fine-grained classification tasks.
- ResNet, with a range of 18 to 152 layers and fewer parameters than VGG (25M for ResNet50), owes its performance to residual blocks. These blocks connect the activation of one layer to a further layer by skipping some in between. Initially designed to address vanishing gradient issues, they also allow skipping layers that may hinder overall performance. Residual blocks share characteristics with LSTM, including gating mechanisms. ResNet models are particularly effective at identifying small details due to their depth.
- EfficientNet was developed with a focus on efficiency and resource optimization while maintaining high accuracy. It achieves this by optimizing a compound coefficient, which determines how much the network scales in width, depth, and resolution. While resource consumption may not be a concern for this project given the dataset's small size, EfficientNet's ability to scale efficiently is undoubtedly advantageous. With a parameter count ranging from 5.3M to 66M, it is smaller or comparable to ResNet.

I adjusted the layer sizes to align with the characteristics of our problem, primarily focusing on binary classification.


**Results:**

<img width="406" alt="Screenshot 2024-02-10 at 9 57 56 AM" src="https://github.com/OrianeCav/Brain-tumor-image-classification/assets/98775053/f573a091-84ba-4fa9-994f-1929350a169f">


Looking at f1-score, the model with the highest performance is the EfficientNet b7, with an f1-score of 85.9%.    
The least performing models are the ResNet types, likely due to their depth, which makes them adept at identifying small details. However, in our case, a more holistic view of the MRI is probably preferable for tumor identification (especially considering that for most of the images, the tumors are at a fairly advanced stage with a big size). The VGG model performs adequately but is surpassed by the EfficientNet. The complexity of VGG may lead to overfitting, whereas the scalability of the EfficientNet makes it a strong candidate for this problem.

### Models created
I started with a simple 1 convolutional layer to have an idea of the baseline accuracy that can be reached easily. Then, I tried a 2 convolutional layers model, and perform hyperparameter optimization (on the batch size, learning rate, number of epochs and number of kernels of the convolutional layers). I used random search with a 3-fold cross validation, as the data set size is fairly small, I wanted to avoid creating a validation set. 
The best f1-score reached was 76.5% by the 2 convolutional layer model.

Then, I used the software weight and biases for better understanding and visualization of the search of the best architecture. 
I started by assessing the performance of 3 architectures and different hyper parameters values. The 3 architectures explored are:
- A "wide" architecture: only one convolutional layer but with a high number of kernels.
- A "deep" architecture: more convolutional layers (3) with a fairly small number of kernels, increasing a bit over the layers.
- A "pyramid" architecture: 3 convolutional layers as well with a high number of kernels on the first one and a decreasing number of kernels over the layers.

The weight and biases project report can be found [here](https://wandb.ai/oriane-cavrois/brain_image_architecture_optimization/reports/Architecture-optimization--Vmlldzo2ODM2Mzk0) and the whole project [here](https://wandb.ai/oriane-cavrois/brain_image_architecture_optimization_3?workspace=user-oriane-cavrois).

The wide architecture clearly dominated in performance and provided the best results. 
So I focused on this architecture to optimize the parameters: the number of kernel and kernel size of the convolutional layer, the kernel size of the maxpool layer and the dropout rate. I also optimized the number of epochs, batch size and learning rate.

The weight and biases project report for this second search can be found [here](https://wandb.ai/oriane-cavrois/brain_image_large_archi_optimization_4?workspace=user-oriane-cavrois) and the whole project [here](https://wandb.ai/oriane-cavrois/brain_image_large_archi_optimization_4/reports/Parameter-optimization-of-the-wide-architecture--Vmlldzo2ODM2MjYz).

**Results:**

<img width="416" alt="Screenshot 2024-02-10 at 11 58 12 AM" src="https://github.com/OrianeCav/Brain-tumor-image-classification/assets/98775053/27c0c0d3-0360-450c-9853-c22f9e28bb04">


The optimized "wide" architecture CNN has a f1 score of 81.4%. It has a particularly good recall score, that means that the number of brain with a tumor wrongly classified is very low. 

## Run the training and test of the models
There is a training/test pipeline created for the best existing and created models. In order to run them, you can run in the terminal: 
```./train_test_created_model.sh``` for the created model and ```./train_test_efficient_net.sh``` for the efficient net b7 which was the best existing model. Before being able to run these files, you might need to run ```chmod +x train_test_created_model.sh``` or ```chmod +x train_test_efficient_net.sh``` before.
These scripts will create the virtual environment with the required packages in case it does not exists, and then use the script train_test_model.py to train the model, compute the accuracy, precision, recall and f1-score on the test set, and print 4 images, their true label and their predictions. 
