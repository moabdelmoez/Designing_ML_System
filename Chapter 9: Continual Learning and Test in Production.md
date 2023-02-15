If monitoring means passively keeping track of the outputs of whatever model is being used, test in production means proactively choosing which model to produce outputs so that we can evaluate it. The goal of both monitoring and test in production is to understand a model’s performance and figure out when to update it. The goal of continual learning is to safely and efficiently automate the update. All of these concepts allow us to design an ML system that is maintainable and adaptable to changing environments.

## Continual Learning

Ideally, you create a replica of the existing model and update this replica on new data, and only replace the existing model with the updated replica if the updated replica proves to be better. The existing model is called the champion model, and the updated replica, the challenger. This is an oversimplification of the process for the sake of understanding. In reality, a company might have multiple challengers at the same time, and handling the failed challenger is a lot more sophisticated than simply discarding it.

![image](https://user-images.githubusercontent.com/37369603/218972886-6dfe3586-1e45-400f-bc90-65fa2e0fbf99.png)

### Stateless Retraining vs Stateful Training
Most companies do stateless retraining—the model is trained from scratch each time. Continual learning means also allowing stateful training—the model continues training on new data. Stateful training is also known as fine-tuning or incremental learning.

![image](https://user-images.githubusercontent.com/37369603/218972981-dc941cb8-b0b4-4a7d-94c2-5817b3c31795.png)

Continual learning is about setting up infrastructure in a way that allows you, a data scientist or ML engineer, to update your models whenever it is needed, whether from scratch or fine-tuning, and to deploy this update quickly.

### Why Continual Learning?

Or why would you need the ability to update your models as fast as you want?

The first use case of continual learning is to combat data distribution shifts, especially when the shifts happen suddenly.

If we could make our models adapt to each user within their visiting session, the models would be able to make accurate, relevant predictions to users even on their first visit. TikTok, for example, has successfully applied continual learning to adapt their recommender system to each user within minutes. You download the app and, after a few videos, TikTok’s algorithms are able to predict with high accuracy what you want to watch next. I don’t think everyone should try to build something as addictive as TikTok, but it’s proof that continual learning can unlock powerful predictive potential.

Ref: https://towardsdatascience.com/why-tiktok-made-its-user-so-obsessive-the-ai-algorithm-that-got-you-hooked-7895bb1ab423

### Continual Learning Challenges

#### Fresh data access challenge

The first challenge is the challenge to get fresh data. If you want to update your model every hour, you need new data every hour.
If your model’s speed iteration is bottlenecked by labeling speed, it’s also possible to speed up the labeling process by leveraging programmatic labeling tools like Snorkel to generate fast labels with minimal human intervention. It might also be possible to leverage crowdsourced labels to quickly annotate fresh data.

#### Evaluation challenge

The biggest challenge of continual learning isn’t in writing a function to continually update your model—you can do that by writing a script! The biggest challenge is in making sure that this update is good enough to be deployed.

It's crucial to thoroughly test each of your model updates to ensure its performance and safety before deploying the updates to a wider audience.

> Note: A model's lineage is a set of associations between a model and all the components that were involved in the creation of that model. A model has relationships with experiments, datasets, container images, and so on.

### How often to update your models

Before attempting to answer that question, we first need to figure out how much gain your model will get from being updated with fresh data. The more gain your model can get from fresher data, the more frequently it should be retrained.

> Note: model iteration (adding a new feature to an existing model architecture or changing the model architecture) and data iteration (same model architecture and features but you refresh this model with new data).

So, as your infrastructure matures and the process of updating a model is partially automated and can be done in a matter of hours, if not minutes, the answer to this question is contingent on the answer to the following question: “How much performance gain would I get from fresher data?” It’s important to run experiments to quantify the value of data freshness to your models.

## Test in Production
 
Test in production don't have to be scary. There are techniques to help you evaluate your models in production (mostly) safely.

### Shadow Deployment

Shadow deployment might be the safest way to deploy your model or any software update. Shadow deployment works as follows:

* Deploy the candidate model in parallel with the existing model.
* For each incoming request, route it to both models to make predictions, but only serve the existing model’s prediction to the user.
* Log the predictions from the new model for analysis purposes.

Only when you’ve found that the new model’s predictions are satisfactory do you replace the existing model with the new model.
This approach has low risk.  However, this technique isn’t always favorable because it’s expensive. It doubles the number of predictions your system has to generate, which generally means doubling your inference compute cost.

### A/B Testing

A/B testing is a way to compare two variants of an object, typically by testing responses to these two variants, and determining which of the two variants is more effective.

A/B testing has become so prevalent that, as of 2017, companies like Microsoft and Google each conduct over 10,000 A/B tests annually.27 It is many ML engineers’ first response to how to evaluate ML models in production. A/B testing works as follows:

* Deploy the candidate model alongside the existing model.
* A percentage of traffic is routed to the new model for predictions; the rest is routed to the existing model for predictions. It’s common for both variants to serve prediction traffic at the same time. However, there are cases where one model’s predictions might affect another model’s predictions—e.g., in ride-sharing’s dynamic pricing, a model’s predicted prices might influence the number of available drivers and riders, which, in turn, influence the other model’s predictions. In those cases, you might have to run your variants alternatively, e.g., serve model A one day and then serve model B the next day.
* Monitor and analyze the predictions and user feedback, if any, from both models to determine whether the difference in the two models’ performance is statistically significant.

Often, in production, you don’t have just one candidate but multiple candidate models. It’s possible to do A/B testing with more than two variants, which means we can have A/B/C testing or even A/B/C/D testing.

### Canary Release

Canary release is a technique to reduce the risk of introducing a new software version in production by slowly rolling out the change to a small subset of users before rolling it out to the entire infrastructure and making it available to everybody.28 In the context of ML deployment, canary release works as follows:
* Deploy the candidate model alongside the existing model. The candidate model is called the canary.
* A portion of the traffic is routed to the candidate model.
* If its performance is satisfactory, increase the traffic to the candidate model. If not, abort the canary and route all the traffic back to the existing model.
* Stop when either the canary serves all the traffic (the candidate model has replaced the existing model) or when the canary is aborted.

Canary releases can be used to implement A/B testing due to the similarities in their setups. However, you can do canary analysis without A/B testing. For example, you don’t have to randomize the traffic to route to each model. A plausible scenario is that you first roll out the candidate model to a less critical market before rolling out to everybody.

As of today, the standard method for testing models in production is A/B testing. With A/B testing, you randomly route traffic to each model for predictions and measure at the end of your trial which model works better. A/B testing is stateless: you can route traffic to each model without having to know about their current performance. You can do A/B testing even with batch prediction.
