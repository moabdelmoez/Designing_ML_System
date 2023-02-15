How to engineer good features is a complex question with no foolproof answers. The best way to learn is through experience: trying out different features and observing how they affect your models’ performance. It’s also possible to learn from experts. I find it extremely useful to read about how the winning teams of Kaggle competitions engineer their features to learn more about their techniques and the considerations they went through.

State-of-the-art model architectures can still perform poorly if they don’t use a good set of features.

The process of choosing what information to use and how to extract this information into a format usable by your ML models is feature engineering. 

## Common Feature Engineering Operations

Because of the importance and the ubiquity of feature engineering in ML projects, there have been many techniques developed to streamline the process. 

### Handling Missing Values

One of the first things you might notice when dealing with data in production is that some values are missing.

<img width="333" alt="image" src="https://user-images.githubusercontent.com/37369603/218469639-d2aef6d7-55e5-4e69-a036-312c2f492008.png">

There are three types of missing values;
#### 1. Missing not at random

This is when the reason a value is missing is because of the true value itself. In this example, we might notice that some respondents didn’t disclose their income. Upon investigation it may turn out that the income of respondents who failed to report tends to be higher than that of those who did disclose. The income values are missing for reasons related to the values themselves.
  
#### 2. Missing at random

This is when the reason a value is missing is not due to the value itself, but due to another observed variable. In this example, we might notice that age values are often missing for respondents of the gender “A,” which might be because the people of gender A in this survey don’t like disclosing their age.
  
#### 3. Missing completely at random

This is when there’s no pattern in when the value is missing.
	
When encountering missing values, you can either fill in the missing values with certain values (imputation) or remove the missing values (deletion).

#### 1. Deletion

One way to delete is column deletion. If a variable has too many missing values, just remove that variable.

The drawback of this approach is that you might remove important information and reduce the accuracy of your model. 
	
Another way to delete is row deletion. If a sample has missing value(s), just remove that sample. This method can work when the missing values are completely at random (MCAR) and the number of examples with missing values is small, such as less than 0.1%. You don’t want to do row deletion if that means 10% of your data samples are removed.

However, removing rows of data can also remove important information that your model needs to make predictions, especially if the missing values are not at random (MNAR).
	
On top of that, removing rows of data can create biases in your model, especially if the missing values are at random (MAR).
	
#### 2. Imputation

Even though deletion is tempting because it’s easy to do, deleting data can lead to losing important information and introduce biases into your model. If you don’t want to delete missing values, you will have to impute them, which means “fill them with certain values.” Deciding which “certain values” to use is the hard part.
	
One common practice is to fill the missing values with their default, like an empty string.

Another common practice is to fill in missing values with the mean, median, or mode (the most common value).
	
> Note: The median is the middle value when a data set is ordered from least to greatest. The mode is the number that occurs most often in a data set.
	
### Scaling

Before inputting features into models, it’s important to scale them to be similar ranges. This process is called feature scaling.

Empirically, I find the range [–1, 1] to work better than the range [0, 1].

In practice, ML models tend to struggle with features that follow a skewed distribution. To help mitigate the skewness, a technique commonly used is log transformation: apply the log function to your feature. 

#### Encoding categorical features

One-hot encoding > assign 1 to the feature and 0 to others [1 0 0 0  ----]

Hashing trick > assign hash value to each feature

#### Feature crossing

Feature crossing is the technique to combine two or more features to generate new features.

It’s less important in neural networks.

#### Discrete and continuous positional embeddings

First introduced to the deep learning community in the paper “Attention Is All You Need”

> Note: An embedding is a vector that represents a piece of data. One of the most common uses of embeddings is word embeddings, where you can represent each word with a vector.

Positional embedding has become a standard data engineering technique for many applications in both computer vision and NLP.

## Data Leakage

Data leakage refers to the phenomenon when a form of the label “leaks” into the set of features used for making predictions, and this same information is not available during inference.

Common causes for data leakage;

### Splitting time-correlated data randomly instead of by time

To prevent future information from leaking into the training process and allowing models to cheat during evaluation, split your data by time, instead of splitting randomly, whenever possible.

![image](https://user-images.githubusercontent.com/37369603/218470360-39b20e6b-95d1-481d-a9a1-e5f22ed51fcf.png)

### Scaling before splitting

To avoid this type of leakage, always split your data first before scaling, then use the statistics from the train split to scale all the splits.

### Filling in missing data with statistics from the test splits
This type of leakage is similar to the type of leakage caused by scaling, and it can be prevented by using only statistics from the train split to fill in missing values in all the splits.

### Poor handling of data duplication before splitting
To avoid this, always check for duplicates before splitting and also after splitting just to make sure. If you oversample your data, do it after splitting.

## Engineering Good Features

There are one factor you might want to consider when evaluating whether a feature is good for a model: importance to the model and generalization to unseen data.

### Feature Importance

There are many different methods for measuring a feature’s importance.

InterpretML is a great open source package that leverages feature importance to help you understand how your model makes predictions.

SHAP (SHapley Additive exPlanations) is great because it not only measures a feature’s importance to an entire model, it also measures each feature’s contribution to a model’s specific prediction.

## Summary

Here is a summary of best practices for feature engineering:

	• Split data by time into train/valid/test splits instead of doing it randomly.
	• If you oversample your data, do it after splitting.
	• Scale and normalize your data after splitting to avoid data leakage.
	• Use statistics from only the train split, instead of the entire data, to scale your features and handle missing values.
	• Understand how your data is generated, collected, and processed. Involve domain experts if possible.
	• Keep track of your data’s lineage.
	• Understand feature importance to your model.
	• Use features that generalize well.
	• Remove no longer useful features from your models.
