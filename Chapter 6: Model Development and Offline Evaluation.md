We’ll move to the ML algorithm part of ML systems. For me, this has always been the most fun step, as it allows me to play around with different algorithms and techniques, even the latest ones. This is also the first step where I can see all the hard work I’ve put into data and feature engineering transformed into a system whose outputs (predictions) I can use to evaluate the success of my effort.

Model development is an iterative process. After each iteration, you’ll want to compare your model’s performance against its performance in previous iterations and evaluate how suitable this iteration is for production.

## Model Development and Training

### Evaluating ML Models

If you had unlimited time and compute power, the rational thing to do would be to try all possible solutions and see what is best for you. However, time and compute power are limited resources, and you have to be strategic about what models you select.

Classical ML algorithms are not going away. Many recommender systems still rely on collaborative filtering and matrix factorization. Tree-based algorithms, including gradient-boosted trees, still power many classification tasks with strict latency requirements.

Even in applications where neural networks are deployed, classic ML algorithms are still being used in tandem. For example, neural networks and decision trees might be used together in an ensemble. A k-means clustering model might be used to extract features to input into a neural network. Vice versa, a pretrained neural network (like BERT or GPT-3) might be used to generate embeddings to input into a logistic regression model.

When selecting a model for your problem, you don’t choose from every possible model out there, but usually focus on a set of models suitable for your problem, for example;
* if your boss tells you to build a system to detect toxic tweets, you know that this is a text classification problem—given a piece of text, classify whether it’s toxic or not—and common models for text classification include naive Bayes, logistic regression, recurrent neural networks, and transformer-based models such as BERT, GPT, and their variants.
* If your client wants you to build a system to detect fraudulent transactions, you know that this is the classic abnormality detection problem—fraudulent transactions are abnormalities that you want to detect—and common algorithms for this problem are many, including k-nearest neighbors, isolation forest, clustering, and neural networks.
	
When considering what model to use, it’s important to consider not only the model’s performance, measured by metrics such as accuracy, F1 score, and log loss, but also its other properties, such as how much data, compute, and time it needs to train, what’s its inference latency, and interpretability.

> Note: To keep up to date with so many new ML techniques and models, I find it helpful to monitor trends at major ML conferences such as NeurIPS, ICLR, and ICML, as well as following researchers whose work has a high signal-to-noise ratio on Twitter.

Without getting into specifics of different algorithms, here are 5 tips that might help you decide what ML algorithms to work on next;

#### 1. Avoid state of the art trap
	
Many business leaders also want to use state-of-the-art models because they want to make their businesses appear cutting edge.

Researchers often only evaluate models in academic settings, which means that a model being state of the art often means that it performs better than existing models on some static datasets. It doesn’t mean that this model will be fast enough or cheap enough for you to implement. It doesn’t even mean that this model will perform better than other models on your data.
	
While it’s essential to stay up to date with new technologies and beneficial to evaluate them for your business, the most important thing to do when solving a problem is finding solutions that can solve that problem. If there’s a solution that can solve your problem that is much cheaper and simpler than state-of-the-art models, use the simpler solution.
	
#### 2. Start with the simplest models

Zen of Python states that “simple is better than complex,” and this principle is applicable to ML as well. 
	
Simplicity serves three purposes. 
* First, simpler models are easier to deploy, and deploying your model early allows you to validate that your prediction pipeline is consistent with your training pipeline. 
* Second, starting with something simple and adding more complex components step-by-step makes it easier to understand your model and debug it. 
* Third, the simplest model serves as a baseline to which you can compare your more complex models.
	
Simplest models are not always the same as models with the least effort. For example, pretrained BERT models are complex, but they require little effort to get started with, especially if you use a ready-made implementation like the one in Hugging Face’s Transformer.
	
#### 3. Avoid human biases in selecting models

There are a lot of human biases in evaluating models. Part of the process of evaluating an ML architecture is to experiment with different features and different sets of hyperparameters to find the best model of that architecture. If an engineer is more excited about an architecture, they will likely spend a lot more time experimenting with it, which might result in better-performing models for that architecture.
	
#### 4. Evaluate good performance now versus good performance later

The best model now does not always mean the best model two months from now. For example, a tree-based model might work better now because you don’t have a ton of data yet, but two months from now, you might be able to double your amount of training data, and your neural network might perform much better.
	
A simple way to estimate how your model’s performance might change with more data is to use learning curves. A learning curve of a model is a plot of its performance—e.g., training loss, training accuracy, validation accuracy—against the number of training samples it uses.
	
While evaluating models, you might want to take into account their potential for improvements in the near future, and how easy/difficult it is to achieve those improvements.
	
#### 5. Evaluate trade-offs

There are many trade-offs you have to make when selecting models. Understanding what’s more important in the performance of your ML system will help you choose the most suitable model.

Examples;
* Trade off between false negative (COVID-19 screening) and false positive (fingerprint unlocking)
* Trade off between compute requirement, accuracy, performance, and interpretability 
		
### Ensembles

One method that has consistently given a performance boost is to use an ensemble of multiple models instead of just an individual model to make predictions. Each model in the ensemble is called a base learner. For example, for the task of predicting whether an email is SPAM or NOT SPAM, you might have three different models. The final prediction for each email is the majority vote of all three models. So if at least two base learners output SPAM, the email will be classified as SPAM.

Ensembling methods are less favored in production because ensembles are more complex to deploy and harder to maintain. However, they are still common for tasks where a small performance boost can lead to a huge financial gain, such as predicting click-through rate for ads.

When creating an ensemble, the less correlation there is among base learners, the better the ensemble will be. Therefore, it’s common to choose very different types of models for an ensemble. For example, you might create an ensemble that consists of one transformer model, one recurrent neural network, and one gradient-boosted tree.

There are three ways to create an ensemble: bagging, boosting, and stacking. In addition to helping boost performance, according to several survey papers, ensemble methods such as boosting and bagging, together with resampling, have shown to help with imbalanced datasets;
	
 #### 1. Bagging
 
Bagging, shortened from bootstrap aggregating, is designed to improve both the training stability and accuracy of ML algorithms. It reduces variance and helps to avoid overfitting.
	
Given a dataset, instead of training one classifier on the entire dataset, you sample with replacement to create different datasets, called bootstraps, and train a classification or regression model on each of these bootstraps. Sampling with replacement ensures that each bootstrap is created independently from its peers.
	
![image](https://user-images.githubusercontent.com/37369603/218471596-48639744-5272-45f4-8a13-8149d6a4747e.png)
	
If the problem is classification, the final prediction is decided by the majority vote of all models.

If the problem is regression, the final prediction is the average of all models’ predictions.
	
A random forest is an example of bagging. A random forest is a collection of decision trees constructed by both bagging and feature randomness, where each tree can pick only from a random subset of features to use.
	
#### 2. Boosting

Boosting is a family of iterative ensemble algorithms that convert weak learners to strong ones.

Each learner in this ensemble is trained on the same set of samples, but the samples are weighted differently among iterations. As a result, future weak learners focus more on the examples that previous weak learners misclassified.
	
An example of a boosting algorithm is XGBoost. XGBoost used to be the algorithm of choice for many winning teams of ML competitions. It’s been used in a wide range of tasks from classification, ranking, to the discovery of the Higgs Boson. However, many teams have been opting for LightGBM, a distributed gradient boosting framework that allows parallel learning, which generally allows faster training on large datasets.
	
### 3. Stacking

Stacking means that you train base learners from the training data then create a meta-learner that combines the outputs of the base learners to output final predictions.
	
Example;

﻿![image](https://user-images.githubusercontent.com/37369603/218471766-2aa67731-95c0-4353-b1a3-8dc551005973.png)
	
### Experiment tracking and versioning

During the model development process, you often have to experiment with many architectures and many different models to choose the best one for your problem. Some models might seem similar to each other and differ in only one hyperparameter—such as one model using a learning rate of 0.003 and another model using a learning rate of 0.002—and yet their performances are dramatically different. It’s important to keep track of all the definitions needed to re-create an experiment and its relevant artifacts. An artifact is a file generated during an experiment—examples of artifacts can be files that show the loss curve, evaluation loss graph, logs, or intermediate results of a model throughout a training process. This enables you to compare different experiments and choose the one best suited for your needs. Comparing different experiments can also help you understand how small changes affect your model’s performance, which, in turn, gives you more visibility into how your model works.

The process of tracking the progress and results of an experiment is called experiment tracking. The process of logging all the details of an experiment for the purpose of possibly recreating it later or comparing it with other experiments is called versioning. These two go hand in hand with each other. Tools such as MLflow and Weights & Biases are used for both (experiment tracking, and versioning).

#### Experiment Tracking

A large part of training an ML model is babysitting the learning processes. Many problems can arise during the training process, including loss not decreasing, overfitting, underfitting, fluctuating weight values, dead neurons, and running out of memory. It’s important to track what’s going on during training not only to detect and address these issues but also to evaluate whether your model is learning anything useful.
	
The majority of people just focus on loss, and speed, following is just a small list of things you might want to consider tracking for each experiment during its training process;
* The loss curve corresponding to the train split and each of the eval splits.
* The model performance metrics that you care about on all non-test splits, such as accuracy, F1, perplexity.
* The log of corresponding sample, prediction, and ground truth label. This comes in handy for ad hoc analytics and sanity check.
* The speed of your model, evaluated by the number of steps per second or, if your data is text, the number of tokens processed per second.
* System performance metrics such as memory usage and CPU/GPU utilization. They’re important to identify bottlenecks and avoid wasting system resources.
* The values over time of any parameter and hyperparameter whose changes can affect your model’s performance, such as the learning rate if you use a learning rate schedule; gradient norms (both globally and per layer), especially if you’re clipping your gradient norms; and weight norm, especially if you’re doing weight decay.
	
In general, tracking gives you observability into the state of your model.
	
#### Versioning

ML systems are part code, part data, so you need to not only version your code but your data as well. Code versioning has more or less become a standard in the industry. However, at this point, data versioning is like flossing. Everyone agrees it’s a good thing to do, but few do it.
	
There are a few reasons why data versioning is challenging. One reason is that because data is often much larger than code, we can’t use the same strategy that people usually use to version code to version data.
	
Here are some of the things that might cause an ML model to fail:
* Theoretical constraints; A model might fail because the data it learns from doesn’t conform to its assumptions. For example, you use a linear model for the data whose decision boundaries aren’t linear.
* Poor implementation of model; The model might be a good fit for the data, but the bugs are in the implementation of the model. For example, if you use PyTorch, you might have forgotten to stop gradient updates during evaluation when you should.
* Poor choice of hyperparameters; With the same model, one set of hyperparameters can give you the state-of-the-art result but another set of hyperparameters might cause the model to never converge. The model is a great fit for your data, and its implementation is correct, but a poor set of hyperparameters might render your model useless.
* Data problems; There are many things that could go wrong in data collection and preprocessing that might cause your models to perform poorly, such as data samples and labels being incorrectly paired, noisy labels, features normalized using outdated statistics, and more.

There is, unfortunately, still no scientific approach to debugging in ML. However, there have been a number of tried-and-true debugging techniques published by experienced ML engineers and researchers.

The following are three of them; Ref: https://karpathy.github.io/2019/04/25/recipe/

##### 1. Start simple and gradually add more components;

Start with the simplest model and then slowly add more components to see if it helps or hurts the performance.
	
##### 2. Overfit a single batch;

After you have a simple implementation of your model, try to overfit a small amount of training data and run evaluation on the same data to make sure that it gets to the smallest possible loss. If it’s for image recognition, overfit on 10 images and see if you can get the accuracy to be 100%, or if it’s for machine translation, overfit on 100 sentence pairs and see if you can get to a BLEU score of near 100. If it can’t overfit a small amount of data, there might be something wrong with your implementation.

##### 3. Set the random seed

Setting a random seed ensures consistency between different runs. It also allows you to reproduce errors and other people to reproduce your results.
	
### Distributed Training

As models are getting bigger and more resource-intensive, companies care a lot more about training at scale.

It’s common to train a model using data that doesn’t fit into memory.
When a sample of your data is large.

#### Data parallelism

It’s now the norm to train ML models on multiple machines. The most common parallelization method supported by modern ML frameworks is data parallelism: you split your data on multiple machines, train your model on all of them, and accumulate gradients. This gives rise to a couple of issues.
	
The below figure explained the difference between async SGD, and sync SGD;
	
![image](https://user-images.githubusercontent.com/37369603/218472547-e878e6c8-9c54-464c-943d-9725a2bbc97a.png)
	
#### Model parallelism

With data parallelism, each worker has its own copy of the whole model and does all the computation necessary for its copy of the model. Model parallelism is when different components of your model are trained on different machines;
	
For example, machine 0 handles the computation for the first two layers while machine 1 handles the next two layers, or some machines can handle the forward pass while several others handle the backward pass.
	
![image](https://user-images.githubusercontent.com/37369603/218472634-7355cde3-08a4-414b-94e0-d0dbc142e083.png)

	
Model parallelism can be misleading because in some cases parallelism doesn’t mean that different parts of the model in different machines are executed in parallel. For example, if your model is a neural network and you put the first layer on machine 1 and the second layer on machine 2, and layer 2 needs outputs from layer 1 to execute, then machine 2 has to wait for machine 1 to finish first to run.
	
Model parallelism and data parallelism aren’t mutually exclusive. Many companies use both methods for better utilization of their hardware, even though the setup to use both methods can require significant engineering effort.
	
### AutoML

AutoML refers to automating the process of finding ML algorithms to solve real-world problems.
	
#### Soft AutoML: Hyperparameter tunning

One mild form, and the most popular form, of AutoML in production is hyperparameter tuning. A hyperparameter is a parameter supplied by users whose value is used to control the learning process, e.g., learning rate, batch size, number of hidden layers, number of hidden units, dropout probability, β1 and β2 in Adam optimizer, etc. Even quantization—e.g., whether to use 32 bits, 16 bits, or 8 bits to represent a number or a mixture of these representations—can be considered a hyperparameter to tune.
		
Popular ML frameworks either come with built-in utilities or have third-party utilities for hyperparameter tuning.
		
#### Hard AutoML: Architecture search, and learned optimizer

Some teams take hyperparameter tuning to the next level: what if we treat other components of a model or the entire model as hyperparameters.
		
Instead of manually putting a pooling layer after a convolutional layer or ReLU (rectified linear unit) after linear, you give your algorithm these building blocks and let it figure out how to combine them. This area of research is known as architectural search, or neural architecture search (NAS) for neural networks, as it searches for the optimal model architecture.
		
> Note: EfficientNet is an AutoML approach from google about compound scaling, it’s used for Image classification.
		
In a typical ML training process, you have a model and then a learning procedure, an algorithm that helps your model find the set of parameters that minimize a given objective function for a given set of data. The most common learning procedure for neural networks today is gradient descent, which leverages an optimizer to specify how to update a model’s weights given gradient updates. Popular optimizers are, as you probably already know, Adam, Momentum, SGD, etc.
	
### Four phases of ML development
	
#### Phase 1: before machine learning

According to Martin Zinkevich in his magnificent “Rules of Machine Learning: Best Practices for ML Engineering”: “If you think that machine learning will give you a 100% boost, then a heuristic will get you 50% of the way there. ”You might even find that non-ML solutions work fine and you don’t need ML yet.
	
#### Phase 2: simplest machine learning models

For your first ML model, you want to start with a simple algorithm, something that gives you visibility into its working to allow you to validate the usefulness of your problem framing and your data. Logistic regression, gradient-boosted trees, k-nearest neighbors can be great for that. They are also easier to implement and deploy, which allows you to quickly build out a framework from data engineering to development to deployment that you can test out and gain confidence on.
	
#### Phase 3: optimizing simple models

Once you have your ML framework in place, you can focus on optimizing the simple ML models with different objective functions, hyperparameter search, feature engineering, more data, and ensembles.
	
#### Phase 4: complex models

Once you’ve reached the limit of your simple models and your use case demands significant model improvement, experiment with more complex models.
	
## Model Evaluation

One common but quite difficult question I often encounter when helping companies with their ML strategies is: “How do I know that our ML models are any good?”

Lacking a clear understanding of how to evaluate your ML systems is not necessarily a reason for your ML project to fail, but it might make it impossible to find the best solution for your need, and make it harder to convince your managers to adopt ML. You might want to partner with the business team to develop metrics for model evaluation that are more relevant to your company’s business.

Once your model is deployed, you’ll need to continue monitoring and testing your model in production.

Evaluation metrics, by themselves, mean little. When evaluating your model, it’s essential to know the baseline you’re evaluating it against. The exact baselines should vary from one use case to another.
Examples;
	i. Random baseline
	ii. Simple heuristic (forget ML)
	iii. Zaro rule baseline (predict the most common class)
	iv. Human baseline
	v. Existing solution

### Evaluation Methods

In academic settings, when evaluating ML models, people tend to fixate on their performance metrics. However, in production, we also want our models to be robust, fair, calibrated, and overall make sense.

#### Perturbation tests

Ideally, the inputs used to develop your model should be similar to the inputs your model will have to work with in production, but it’s not possible in many cases. 
	
The inputs your models have to work with in production are often noisy compared to inputs in development. The model that performs best on training data isn’t necessarily the model that performs best on noisy data.
	
Perturbation means adding noise, usually to the training data to simulate the real data.
	
#### Invariance tests

Certain changes to the inputs shouldn’t lead to changes in the output. In the preceding case, changes to race information shouldn’t affect the mortgage outcome. Similarly, changes to applicants’ names shouldn’t affect their resume screening results nor should someone’s gender affect how much they should be paid. If these happen, there are biases in your model, which might render it unusable no matter how good its performance is.
	
#### Directional expectation tests

Certain changes to the inputs should, however, cause predictable changes in outputs. For example, when developing a model to predict housing prices, keeping all the features the same but increasing the lot size shouldn’t decrease the predicted price, and decreasing the square footage shouldn’t increase it. If the outputs change in the opposite expected direction, your model might not be learning the right thing, and you need to investigate it further before deploying it.
