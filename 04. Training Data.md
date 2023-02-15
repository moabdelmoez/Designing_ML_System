Despite the importance of training data in developing and improving ML models, ML curricula are heavily skewed toward modeling, which is considered by many practitioners the “fun” part of the process. Building a state-of-the-art model is interesting. Spending days wrangling with a massive amount of malformatted data that doesn’t even fit into your machine’s memory is frustrating.

Data is messy, complex, unpredictable, and potentially treacherous. If not handled properly, it can easily sink your entire ML operation. But this is precisely the reason why data scientists and ML engineers should learn how to handle data well, saving us time and headache down the road.

Like other steps in building ML systems, creating training data is an iterative process. As your model evolves through a project lifecycle, your training data will likely also evolve.

Before we move forward, I just want to echo a word of caution that has been said many times yet is still not enough. Data is full of potential biases. These biases have many possible causes. There are biases caused during collecting, sampling, or labeling. Historical data might be embedded with human biases, and ML models, trained on this data, can perpetuate them. Use data but don’t trust it too much!

## Sampling

Sampling is an integral part of the ML workflow that is, unfortunately, often overlooked in typical ML coursework. Sampling happens in many steps of an ML project lifecycle;
* such as sampling from all possible real-world data to create training data,
* sampling from a given dataset to create splits for training, validation, and testing; or
* sampling from all possible events that happen within your ML system for monitoring purposes. 
	
In many other cases, sampling is helpful as it allows you to accomplish a task faster and cheaper. For example, when considering a new model, you might want to do a quick experiment with a small subset of your data to see if the new model is promising first before training this new model on all your data.

There are two families of sampling: nonprobability sampling and random sampling. 

### 1. Nonprobability sampling

Nonprobability sampling is when the selection of data isn’t based on any probability criteria.
The samples selected by nonprobability criteria are not representative of the real-world data and therefore are riddled with selection biases.
Nonprobability sampling can be a quick and easy way to gather your initial data to get your project off the ground. However, for reliable models, you might want to use probability-based sampling.

### 2. Simple Random Sampling

In the simplest form of random sampling, you give all samples in the population equal probabilities of being selected. For example, you randomly select 10% of the population, giving all members of this population an equal 10% chance of being selected.
The advantage of this method is that it’s easy to implement. The drawback is that rare categories of data might not appear in your selection.

## Labeling

Despite the promise of unsupervised ML, most ML models in production today are supervised, which means that they need labeled data to learn from. The performance of an ML model still depends heavily on the quality and quantity of the labeled data it’s trained on.

### Hand Labels

First, hand-labeling data can be expensive, especially if subject matter expertise is required.

Second, hand labeling poses a threat to data privacy. Hand labeling means that someone has to look at your data, which isn’t always possible if 
your data has strict privacy requirements.

Third, hand labeling is slow. Slow labeling leads to slow iteration speed and makes your model less adaptive to changing environments and requirements.

If the task changes or data changes, you’ll have to wait for your data to be relabeled before updating your model. The longer the process takes, the more your existing model performance will degrade.

### Label Multiplicity

Often, to obtain enough labeled data, companies have to use data from multiple sources and rely on multiple annotators who have different levels of expertise. These different data sources and annotators also have different levels of accuracy. This leads to the problem of label ambiguity or label multiplicity: what to do when there are multiple conflicting labels for a data instance.
To minimize the disagreement among annotators, it’s important to first have a clear problem definition. 

### Data Linage

Indiscriminately using data from multiple sources, generated with different annotators, without examining their quality can cause your model to fail mysteriously.

It’s good practice to keep track of the origin of each of your data samples as well as its labels, a technique known as data lineage. Data lineage helps you both flag potential biases in your data and debug your models. For example, if your model fails mostly on the recently acquired data samples, you might want to look into how the new data was acquired. On more than one occasion, we’ve discovered that the problem wasn’t with our model, but because of the unusually high number of wrong labels in the data that we’d acquired recently.

### Natural Labels

Hand-labeling isn’t the only source for labels. You might be lucky enough to work on tasks with natural ground truth labels. Tasks with natural labels are tasks where the model’s predictions can be automatically evaluated or partially evaluated by the system. An example is the model that estimates time of arrival for a certain route on Google Maps. If you take that route, by the end of your trip, Google Maps knows how long the trip actually took, and thus can evaluate the accuracy of the predicted time of arrival. Another example is stock price prediction. If your model predicts a stock’s price in the next two minutes, then after two minutes, you can compare the predicted price with the actual price.

### Feedback Loop Length
There are two types, short feedback loop like click-ads. And long feedback loop like fraud detection.

### Handling the lack of labels
	
<img width="625" alt="image" src="https://user-images.githubusercontent.com/37369603/218468526-bf943f60-1c31-4d1b-a2a7-73ad22585be4.png">

#### 1. Weak Supervision

The insight behind weak supervision is that people rely on heuristics, which can be developed with subject matter expertise, to label data. 

Libraries like Snorkel are built around the concept of a labeling function (LF): a function that encodes heuristics. The preceding heuristics can be expressed by the following function:

<img width="455" alt="image" src="https://user-images.githubusercontent.com/37369603/218468592-1ab50ae2-a5b8-42cb-9af7-718ca6c82a13.png">

It’s important to combine, denoise, and reweight all LFs to get a set of most likely to be correct labels.

In theory, you don’t need any hand labels for weak supervision. However, to get a sense of how accurate your LFs are, a small number of hand labels is recommended. These hand labels can help you discover patterns in your data to write better LFs.

Weak supervision is a simple but powerful paradigm. However, it’s not perfect. In some cases, the labels obtained by weak supervision might be too noisy to be useful. But even in these cases, weak supervision can be a good way to get you started when you want to explore the effectiveness of ML without wanting to invest too much in hand labeling up front.

#### 2. Semi Supervision

If weak supervision leverages heuristics to obtain noisy labels, semi-supervision leverages structural assumptions to generate new labels based on a small set of initial labels. Unlike weak supervision, semi-supervision requires an initial set of labels.

A classic semi-supervision method is self-training. You start by training a model on your existing set of labeled data and use this model to make predictions for unlabeled samples. Assuming that predictions with high raw probability scores are correct, you add the labels predicted with high probability to your training set and train a new model on this expanded training set. This goes on until you’re happy with your model performance.

In most cases, the similarity can only be discovered by more complex methods. For example, you might need to use a clustering method or a k-nearest neighbors algorithm to discover samples that belong to the same cluster.

> Note: Perturbation means adding noise, usually to the training data.

### 3. Transfer Learning

Transfer learning refers to the family of methods where a model developed for a task is reused as the starting point for a model on a second task.

In some cases, such as in zero-shot learning scenarios, you might be able to use the base model on a downstream task directly. In many cases, you might need to fine-tune the base model. Fine-tuning means making small changes to the base model, such as continuing to train the base model or a part of the base model on data from a given downstream task.

Transfer learning is especially appealing for tasks that don’t have a lot of labeled data. Even for tasks that have a lot of labeled data, using a pretrained model as the starting point can often boost the performance significantly compared to training from scratch.

Transfer learning also lowers the entry barriers into ML, as it helps reduce the up-front cost needed for labeling data to build ML applications.

### 4. Active Learning

Active learning is a method for improving the efficiency of data labels. The hope here is that ML models can achieve greater accuracy with fewer training labels if they can choose which data samples to learn from.

Instead of randomly labeling data samples, you label the samples that are most helpful to your models according to some metrics or heuristics.

The data can come from the real-world distribution where you have a stream of data coming in, as in production, and your model chooses samples from this stream of data to label.

## Class Imbalance

Class imbalance typically refers to a problem in classification tasks where there is a substantial difference in the number of samples in each class of the training data. For example, in a training dataset for the task of detecting lung cancer from X-ray images, 99.99% of the X-rays might be of normal lungs, and only 0.01% might contain cancerous cells.

Class imbalance can also happen with regression tasks where the labels are continuous.

### Handling Class Imbalance
We will cover 3 approaches to handling class imbalance;

#### 1. Using the right evaluation metrics

The most important thing to do when facing a task with class imbalance is to choose the appropriate evaluation metrics. Wrong metrics will give you the wrong ideas of how your models are doing and, subsequently, won’t be able to help you develop or choose models good enough for your task.

The overall accuracy and error rate are the most frequently used metrics to report the performance of ML models. However, these are insufficient metrics for tasks with class imbalance because they treat all classes equally, which means the performance of your model on the majority class will dominate these metrics. This is especially bad when the majority class isn’t what you care about.

F1, precision, and recall are metrics that measure your model’s performance with respect to the positive class in binary classification problems, as they rely on true positive—an outcome where the model correctly predicts the positive class.

<img width="782" alt="image" src="https://user-images.githubusercontent.com/37369603/218468857-c1665467-5226-4753-861a-559305c57ef2.png">

![image](https://user-images.githubusercontent.com/37369603/218468885-af96d7f9-9262-47b6-aeb5-c5c724c7f513.png)

#### 2. Data-level methods: Resampling

Data-level methods modify the distribution of the training data to reduce the level of imbalance to make it easier for the model to learn. A common family of techniques is resampling. Resampling includes oversampling, adding more instances from the minority classes, and undersampling, removing instances of the majority classes. The simplest way to undersample is to randomly remove instances from the majority class, whereas the simplest way to oversample is to randomly make copies of the minority class until you have a ratio that you’re happy with.

A popular method of oversampling low-dimensional data is SMOTE (synthetic minority oversampling technique).

#### 3. Algorithm-level methods

If data-level methods mitigate the challenge of class imbalance by altering the distribution of your training data, algorithm-level methods keep the training data distribution intact but alter the algorithm to make it more robust to class imbalance.

Because the loss function (or the cost function) guides the learning process, many algorithm-level methods involve adjustment to the loss function. The key idea is that if there are two instances, x1 and x2, and the loss resulting from making the wrong prediction on x1 is higher than x2, the model will prioritize making the correct prediction on x1 over making the correct prediction on x2. By giving the training instances we care about higher weight, we can make the model focus more on learning these instances.

## Data Augmentation 

Data augmentation is a family of techniques that are used to increase the amount of training data. Traditionally, these techniques are used for tasks that have limited training data, such as in medical imaging. However, in the last few years, they have shown to be useful even when we have a lot of data—augmented data can make our models more robust to noise and even adversarial attacks.

There are three main types of data augmentation;

### 1. Simple label-preserving transformations

In computer vision, the simplest data augmentation technique is to randomly modify an image while preserving its label. You can modify the image by cropping, flipping, rotating, inverting (horizontally or vertically), erasing part of the image, and more. 

In NLP, you can randomly replace a word with a similar word, assuming that this replacement wouldn’t change the meaning or the sentiment of the sentence

### 2. Perturbation

Perturbation is also a label-preserving operation, but because sometimes it’s used to trick models into making wrong predictions, I thought it deserves its own section.

Neural networks, in general, are sensitive to noise. In the case of computer vision, this means that adding a small amount of noise to an image can cause a neural network to misclassify it. Su et al. showed that 67.97% of the natural images in the Kaggle CIFAR-10 test dataset and 16.04% of the ImageNet test images can be misclassified by changing just one pixel.

Adding noisy samples to training data can help models recognize the weak spots in their learned decision boundary and improve their performance.

### 3. Data Synthesis

Since collecting data is expensive and slow, with many potential privacy concerns, it’d be a dream if we could sidestep it altogether and train our models with synthesized data. Even though we’re still far from being able to synthesize all training data, it’s possible to synthesize some training data to boost a model’s performance.
