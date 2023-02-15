Developing this logic requires both ML knowledge and SME.

![image](https://user-images.githubusercontent.com/37369603/218969761-8ed006b2-7dab-4b7e-9704-b91dad99e460.png)

“Deploy” is a loose term that generally means making your model running and accessible.

Note: Exporting a model means converting this model into a format that can be used by another application. Some people call this process “serialization.” There are two parts of a model that you can export: the model definition and the model’s parameter values. The model definition defines the structure of your model, such as how many hidden layers it has and how many units in each layer. The parameter values provide the values for these units and layers. Usually, these two parts are exported together.

## Machine Learning Deployment Myths

* Myth 1: you only deploy one or two ML models at a time
* Myth 2: if you don't do anything, Model performance remains the same
* Myth 3: you won't need to update your models as much
* Myth 4: most ML engineers don't need to worry about scale

## Batch Prediction vs Online Prediction

One fundamental decision you’ll have to make that will affect both your end users and developers working on your system is how it generates and serves its predictions to end users: online or batch.

### Online Prediction

Online prediction is when predictions are generated and returned as soon as requests for these predictions arrive. For example, you enter an English sentence into Google Translate and get back its French translation immediately.
When prediction requests are sent via HTTP requests, online prediction is also known as synchronous prediction: predictions are generated in synchronization with requests.

![image](https://user-images.githubusercontent.com/37369603/218969948-f125af29-c4ef-4557-9b93-3201c2ad5f03.png)

To overcome the latency challenge of online prediction, two components are required:

  * A (near) real-time pipeline that can work with incoming data, extract streaming features (if needed), input them into a model, and return a prediction in near real time. A streaming pipeline with real-time transport and a stream computation engine can help with that.
	* A model that can generate predictions at a speed acceptable to its end users. For most consumer apps, this means milliseconds.

### Batch Prediction

Batch prediction is when predictions are generated periodically or whenever triggered. The predictions are stored somewhere, such as in SQL tables or an in-memory database, and retrieved as needed. For example, Netflix might generate movie recommendations for all of its users every four hours, and the precomputed recommendations are fetched and shown to users when they log on to Netflix. Batch prediction is also known as asynchronous prediction: predictions are generated asynchronously with requests.

![image](https://user-images.githubusercontent.com/37369603/218970109-a0c24957-70c8-4003-b1fc-26ca8cb4ab53.png)

Batch prediction is good for when you want to generate a lot of predictions and don’t need the results immediately. You don’t have to use all the predictions generated. For example, you can make predictions for all customers on how likely they are to buy a new product, and reach out to the top 10%.

> Note: features computed from historical data, such as data in databases and data warehouses, are batch features. Features computed from streaming data—data in real-time transports—are streaming features.

![image](https://user-images.githubusercontent.com/37369603/218970157-24845f90-b8eb-40d5-a29b-e3aee3954914.png)

> Note: To people coming to ML from an academic background, the more natural way to serve predictions is probably online. You give your model an input and it generates a prediction as soon as it receives that input. This is likely how most people interact with their models while prototyping. This is also likely easier to do for most companies when first deploying a model. You export your model, upload the exported model to Amazon SageMaker or Google App Engine, and get back an exposed endpoint. Now, if you send a request that contains an input to that endpoint, it will send back a prediction generated on that input.

## Model Compression

If the model you want to deploy takes too long to generate predictions, there are three main approaches to reduce its inference latency: make it do inference faster, make the model smaller, or make the hardware it’s deployed on run faster.

The process of making a model smaller is called model compression, and the process to make it do inference faster is called inference optimization.
The number of research papers on model compression is growing. Off-the-shelf utilities are proliferating. As of April 2022, Awesome Open Source has a list of “The Top 168 Model Compression Open Source Projects”, and that list is growing.

The four types of techniques that you might come across the most often are low-rank optimization, knowledge distillation, pruning, and quantization.

### Low-Rank Factorization

> Note: Convolution provides a way of multiplying together  two arrays of numbers, generally of different sizes, but of the same dimensionality, to produce a third array of numbers of the same dimensionality.

The key idea behind low-rank factorization is to replace high-dimensional tensors with lower-dimensional tensors. One type of low-rank factorization is compact convolutional filters, where the over-parameterized (having too many parameters) convolution filters are replaced with compact blocks to both reduce the number of parameters and increase speed.

This method has been used to develop smaller models with significant acceleration compared to standard models. However, it tends to be specific to certain types of models (e.g., compact convolutional filters are specific to convolutional neural networks) and requires a lot of architectural knowledge to design, so it’s not widely applicable to many use cases yet.

### Knowledge Distillation

> Note: Knowledge distillation helps overcome these challenges by capturing and “distilling” the knowledge in a complex machine learning model or an ensemble of models into a smaller single model that is much easier to deploy without significant loss in performance.
https://neptune.ai/blog/knowledge-distillation.

Knowledge distillation is a method in which a small model (student) is trained to mimic a larger model or ensemble of models (teacher). The smaller model is what you’ll deploy.

Even though the student is often trained after a pretrained teacher, both may also be trained at the same time. One example of a distilled network used in production is DistilBERT, which reduces the size of a BERT model by 40% while retaining 97% of its language understanding capabilities and being 60% faster.

The advantage of this approach is that it can work regardless of the architectural differences between the teacher and the student networks. For example, you can get a random forest as the student and a transformer as the teacher.

The disadvantage of this approach is that it’s highly dependent on the availability of a teacher network. If you use a pretrained model as the teacher model, training the student network will require less data and will likely be faster. However, if you don’t have a teacher available, you’ll have to train a teacher network before training a student network, and training a teacher network will require a lot more data and take more time to train.

This method is also sensitive to applications and model architectures, and therefore hasn’t found wide usage in production.

> Note: Most weights in a neural network are useless, so we can remove or reduce most of the weights in a neural network with limited effect on loss - - - by Sparsification.

Pruning and Quantization are examples of sparsification.

### Pruning

Pruning, in the context of neural networks, has two meanings. One is to remove entire nodes of a neural network, which means changing its architecture and reducing its number of parameters. The more common meaning is to find parameters least useful to predictions and set them to 0. In this case, pruning doesn’t reduce the total number of parameters, only the number of nonzero parameters. The architecture of the neural network remains the same. This helps with reducing the size of a model because pruning makes a neural network more sparse, and sparse architecture tends to require less storage space than dense structure. 

Experiments show that pruning techniques can reduce the nonzero parameter counts of trained networks by over 90%, decreasing storage requirements and improving computational performance of inference without compromising overall accuracy.

![image](https://user-images.githubusercontent.com/37369603/218970552-ad619b31-f8d5-4154-9a69-ba6fe244bd44.png)


### Quantization

Quantization reduces a model’s size by using fewer bits to represent its parameters. By default, most software packages use 32 bits to represent a float number (single precision floating point). If a model has 100M parameters and each requires 32 bits to store, it’ll take up 400 MB. If we use 16 bits to represent a number, we’ll reduce the memory footprint by half. Using 16 bits to represent a float is called half precision.

Instead of using floats, you can have a model entirely in integers; each integer takes only 8 bits to represent. This method is also known as “fixed point.”

Quantization not only reduces memory footprint but also improves the computation speed. First, it allows us to increase our batch size. Second, less precision speeds up computation, which further reduces training time and inference latency. Consider the addition of two numbers. If we perform the addition bit by bit, and each takes x nanoseconds, it’ll take 32x nanoseconds for 32-bit numbers but only 16x nanoseconds for 16-bit numbers.

There are downsides to quantization. Reducing the number of bits to represent your numbers means that they decrease the accuracy.

Quantization can either happen during training (quantization aware training), where models are trained in lower precision, or post-training, where models are trained in single-precision floating point and then quantized for inference. Using quantization during training means that you can use less memory for each parameter, which allows you to train larger models on the same hardware.

Recently, low-precision (like FP16, Bfloat16) training has become increasingly popular, with support from most modern training hardware. NVIDIA introduced Tensor Cores, processing units that support mixed-precision training. Google TPUs (tensor processing units) also support training with Bfloat16. Training in fixed-point (INT8) is not yet as popular but has had a lot of promising results.

Fixed-point (INT8) inference has become a standard in the industry. Some edge devices only support fixed-point inference. Most popular frameworks for on-device ML inference—Google’s TensorFlow Lite, Facebook’s PyTorch Mobile, NVIDIA’s TensorRT—offer post-training quantization for free with a few lines of code, Intel’s OpenVINO as well.

## ML on the Cloud and on the Edge

Many companies started their ML deployment in the cloud, Cloud services have done an incredible job to make it easy for companies to bring ML models into production.
However, there are many downsides to cloud deployment. The most major one is cost.

From that Edge computing has many advantages over the cloud, such as;
* Increase Privacy
* Worry-less about network latency
* No dependency on Internet connection
* Reduce cost

To move computation to the edge, the edge devices have to be powerful enough to handle the computation, have enough memory to store ML models and load them into memory, as well as have enough battery or be connected to an energy source to power the application for a reasonable amount of time.
Compiling and Optimizing Models for Edge Devices

For a model built with a certain framework, such as TensorFlow or PyTorch, to run on a hardware backend, that framework has to be supported by the hardware vendor. For example, even though TPUs were released publicly in February 2018, it wasn’t until September 2020 that PyTorch was supported on TPUs. Before then, if you wanted to use a TPU, you’d have to use a framework that TPUs supported.

Providing support for a framework on a hardware backend is time-consuming and engineering-intensive. Mapping from ML workloads to a hardware backend requires understanding and taking advantage of that hardware’s design, and different hardware backends have different memory layouts and compute primitives.

Because of this challenge, framework developers tend to focus on providing support to only a handful of server-class hardware, and hardware vendors tend to offer their own kernel libraries for a narrow range of frameworks. Deploying ML models to new hardware requires significant manual effort.
Instead of targeting new compilers and libraries for every new hardware backend, what if we create a middleman to bridge frameworks and platforms? Framework developers will no longer have to support every type of hardware; they will only need to translate their framework code into this middleman. 

Hardware vendors can then support one middleman instead of multiple frameworks.
This type of “middleman” is called an intermediate representation (IR). IRs lie at the core of how compilers work. From the original code for a model, compilers generate a series of high- and low-level IRs before generating the code native to a hardware backend so that it can run on that hardware backend.

![image](https://user-images.githubusercontent.com/37369603/218970836-788221ad-6ce7-46bc-9430-a198acc35b9d.png)

> Note: This process is also called lowering, as in you “lower” your high-level framework code into low-level hardware-native code. It’s not translating because there’s no one-to-one mapping between them.

High-level IRs are usually computation graphs of your ML models. A computation graph is a graph that describes the order in which your computation is executed. Readers interested can read about computation graphs in PyTorch and TensorFlow.

### Model Optimization

After you’ve “lowered” your code to run your models into the hardware of your choice, an issue you might run into is performance.
In many companies, what usually happens is that data scientists and ML engineers develop models that seem to be working fine in development. However, when these models are deployed, they turn out to be too slow.

Optimizing compilers (compilers that also optimize your code) are an awesome solution, as they can automate the process of optimizing models. In the process of lowering ML model code into machine code, compilers can look at the computation graph of your ML model and the operators it consists of—convolution, loops, cross-entropy—and find a way to speed it up.

There are two ways to optimize your ML models: locally and globally. Locally is when you optimize an operator or a set of operators of your model. Globally is when you optimize the entire computation graph end to end.

There are standard local optimization techniques that are known to speed up your model, most of them making things run in parallel or reducing memory access on chips. Here are four of the common techniques:

#### Vectorization
Given a loop or a nested loop, instead of executing it one item at a time, execute multiple elements contiguous in memory at the same time to reduce latency caused by data I/O.
#### Parallelization
Given an input array (or n-dimensional array), divide it into different, independent work chunks, and do the operation on each chunk individually.
#### Loop tiling
Change the data accessing order in a loop to leverage hardware’s memory layout and cache. This kind of optimization is hardware dependent. A good access pattern on CPUs is not a good access pattern on GPUs.
#### Operator Fusion
	Fuse multiple operators into one to avoid redundant memory access. For example, two operations on the same array require two loops over that array. In a fused case, it’s just one loop.

### Using ML to optimize ML models

To obtain a much bigger speedup, you’d need to leverage higher-level structures of your computation graph. For example, a convolution neural network with the computation graph can be fused vertically or horizontally to reduce memory access and speed up the model.
As hinted by the previous section with the vertical and horizontal fusion for a convolutional neural network, there are many possible ways to execute a given computation graph. For example, given three operators A, B, and C, you can fuse A with B, fuse B with C, or fuse A, B, and C all together.

Traditionally, framework and hardware vendors hire optimization engineers who, based on their experience, come up with heuristics on how to best execute the computation graph of a model. For example, NVIDIA might have an engineer or a team of engineers who focus exclusively on how to make ResNet-50 run really fast on their DGX A100 server.

Hardware vendors like NVIDIA and Google focus on optimizing popular models like ResNet-50 and BERT for their hardware. But what if you, as an ML researcher, come up with a new model architecture? You might need to optimize it yourself to show that it’s fast first before it’s adopted and optimized by hardware vendors.

What if we use ML to narrow down the search space so we don’t have to explore that many paths, and predict how long a path will take so that we don’t have to wait for the entire computation graph to finish executing?

Examples;
* cuDNN autotune searches over a predetermined set of options to execute a convolution operator and then chooses the fastest way. cuDNN autotune, despite its effectiveness, only works for convolution operators.
* autoTVM, which is part of the open source compiler stack TVM. autoTVM works with subgraphs instead of just an operator, so the search spaces it works with are much more complex.

While the results of ML-powered compilers are impressive, they come with a catch: they can be slow. You go through all the possible paths and find the most optimized ones. This process can take hours, even days for complex ML models. However, it’s a one-time operation, and the results of your optimization search can be cached and used to both optimize existing models and provide a starting point for future tuning sessions. You optimize your model once for one hardware backend then run it on multiple devices of that same hardware type. This sort of optimization is ideal when you have a model ready for production and target hardware to run inference on.

### ML in Browsers

When talking about browsers, many people think of JavaScript. There are tools that can help you compile your models into JavaScript, such as TensorFlow.js, Synaptic, and brain.js. However, JavaScript is slow, and its capacity as a programming language is limited for complex logics such as extracting features from data.

A more promising approach is WebAssembly (WASM). WASM is an open standard that allows you to run executable programs in browsers. After you’ve built your models in scikit-learn, PyTorch, TensorFlow, or whatever frameworks you’ve used, instead of compiling your models to run on specific hardware, you can compile your model to WASM. You get back an executable file that you can just use with JavaScript.

The main drawback of WASM is that because WASM runs in browsers, it’s slow. Even though WASM is already much faster than JavaScript, it’s still slow compared to running code natively on devices (such as iOS or Android apps).
