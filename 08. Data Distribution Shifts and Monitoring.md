A model’s performance degrades over time in production. Once a model has been deployed, we still have to continually monitor its performance to detect issues as well as deploy updates to fix these issues.

## Causes of ML System Failures

Before we identify the cause of ML system failures, let’s briefly discuss what an ML system failure is. A failure happens when one or more expectations of the system is violated.

For an ML system, we care about both its operational metrics and its ML performance metrics.

For example, consider an English-French machine translation system. Its operational expectation might be that, given an English sentence, the system returns a French translation within a one-second latency. Its ML performance expectation is that the returned translation is an accurate translation of the original English sentence 99% of the time. If you enter an English sentence into the system and don’t get back a translation, the first expectation is violated, so this is a system failure.

Operational expectation violations are easier to detect, as they’re usually accompanied by an operational breakage such as a timeout, a 404 error on a webpage, an out-of-memory error, or a segmentation fault. However, ML performance expectation violations are harder to detect as doing so requires measuring and monitoring the performance of ML models in production.

For this reason, we say that ML systems often fail silently.

Let’s examine two types of failures: software failures, and ML-specific failures.

### Software System Failures

Just because some failures are not specific to ML doesn’t mean they’re not important for ML engineers to understand.
There are many examples for software failures;
* Dependency failure
* Deployment failure
* Hardware failure
* Downtime or crashing

A reason for the prevalence of software system failures is that because ML adoption in the industry is still nascent, tooling around ML production is limited and best practices are not yet well developed or standardized. However, as toolings and best practices for ML production mature, there are reasons to believe that the proportion of software system failures will decrease and the proportion of ML-specific failures will increase.

### ML-Specific Failures

ML-specific failures are failures specific to ML systems. Examples include data collection and processing problems, poor hyperparameters, changes in the training pipeline not correctly replicated in the inference pipeline and vice versa, data distribution shifts that cause a model’s performance to deteriorate over time, edge cases, and degenerate feedback loops.

We'll discuss two new but very common problems that arise after a model has been deployed: production data differing from training data, edge cases..

#### Production data differing from training data

A common failure mode is that a model does great when first deployed, but its performance degrades over time as the data distribution changes. This failure mode needs to be continually monitored and detected for as long as a model remains in production.
		
#### Edge Cases

An ML model that performs well on most cases but fails on a small number of cases might not be usable if these failures cause catastrophic consequences.

> Note: Outliers refer to data: an example that differs significantly from other examples. Edge cases refer to performance: an example where a model performs significantly worse than other examples. An outlier can cause a model to perform unusually poorly, which makes it an edge case. However, not all outliers are edge cases.
#### Data Distribution Shifts

Data distribution shift refers to the phenomenon in supervised learning when the data a model works with changes over time, which causes this model’s predictions to become less accurate as time passes. The distribution of the data the model is trained on is called the source distribution. The distribution of the data the model runs inference on is called the target distribution.

Data distribution shifts are only a problem if they cause your model’s performance to degrade. So the first idea might be to monitor your model’s accuracy-related metrics—accuracy, F1 score, recall, AUC-ROC, etc.—in production to see whether they have changed. “Change” here usually means “decrease,” but if my model’s accuracy suddenly goes up or fluctuates significantly for no reason that I’m aware of, I’d want to investigate.

Many companies assume that data shifts are inevitable, so they periodically retrain their models—once a month, once a week, or once a day—regardless of the extent of the shift. How to determine the optimal frequency to retrain your models is an important decision that many companies still determine based on gut feelings instead of experimental data.

To make a model work with a new distribution in production, There’s an approach that is usually done in the industry today: retrain your model using the labeled data from the target distribution. However, retraining your model is not so straightforward. Retraining can mean retraining your model from scratch on both the old and new data or continuing training the existing model on new data. The latter approach is also called fine-tuning.

## Monitoring and Observability 

> Note: While monitoring alerts the team to a potential issue, observability helps the team detect and solve the root cause of the issue.

Monitoring and observability are sometimes used exchangeably, but they are different. Monitoring refers to the act of tracking, measuring, and logging different metrics that can help us determine when something goes wrong. Observability means setting up our system in a way that gives us visibility into our system to help us investigate what went wrong.

Monitoring is all about metrics. Because ML systems are software systems, the first class of metrics you’d need to monitor are the operational metrics.
These metrics are designed to convey the health of your systems. They are generally divided into three levels: the network the system is run on, the machine the system is run on, and the application that the system runs.

Examples of these metrics are latency; throughput; the number of prediction requests your model receives in the last minute, hour, day; the percentage of requests that return with a 2xx code; CPU/GPU utilization; memory utilization; etc. No matter how good your ML model is, if the system is down, you’re not going to benefit from it.

However, for ML systems, the system health extends beyond the system uptime. If your ML system is up but its predictions are garbage, your users aren’t going to be happy. Another class of metrics you’d want to monitor are ML-specific metrics that tell you the health of your ML models.

### ML-specific Metrics

Within ML-specific metrics, there are generally four artifacts to monitor: a model’s accuracy-related metrics, predictions, features, and raw inputs. These are artifacts generated at four different stages of an ML system pipeline.

![image](https://user-images.githubusercontent.com/37369603/218972490-004f277c-71b3-4092-9b7f-3ae97f65a63f.png)

When something goes wrong with an observable system, we should be able to figure out what went wrong by looking at the system’s logs and metrics without having to ship new code to the system. Observability is about instrumenting your system in a way to ensure that sufficient information about a system’s runtime is collected and analyzed.

In ML, observability encompasses interpretability. Interpretability helps us understand how an ML model works, and observability helps us understand how the entire ML system, which includes the ML model, works. For example, when a model’s performance degrades over the last hour, being able to interpret which feature contributes the most to all the wrong predictions made over the last hour will help with figuring out what went wrong with the system and how to fix it.

Monitoring Toolbox;
* Logs
* Dashboards
* Alerts
