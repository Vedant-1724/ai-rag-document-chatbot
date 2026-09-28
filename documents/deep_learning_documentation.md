# Deep Learning Documentation

> This document was collected from official documentation sources for use as a Retrieval-Augmented Generation (RAG) corpus.

## Sources


- [PyTorch Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html)
- [PyTorch Quickstart](https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html)
- [PyTorch Tensors](https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html)
- [Datasets and DataLoaders](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html)
- [Building the Neural Network](https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html)
- [Automatic Differentiation](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)
- [Optimization](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)
- [Save and Load the Model](https://docs.pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html)


---


## PyTorch Learn the Basics

**Source:** https://docs.pytorch.org/tutorials/beginner/basics/intro.html

Rate this Page
★
★
★
★
★
beginner/basics/intro
Run in Google Colab
Colab
Download Notebook
Notebook
View on GitHub
GitHub
Note
Go to the end
to download the full example code.
Learn the Basics
||
Quickstart
||
Tensors
||
Datasets & DataLoaders
||
Transforms
||
Build Model
||
Autograd
||
Optimization
||
Save & Load Model
Learn the Basics
#
Created On: Feb 09, 2021 | Last Updated: Jan 20, 2026 | Last Verified: Nov 05, 2024
Authors:
Suraj Subramanian
,
Seth Juarez
,
Cassie Breviu
,
Dmitry Soshnikov
,
Ari Bornstein
Most machine learning workflows involve working with data, creating models, optimizing model
parameters, and saving the trained models. This tutorial introduces you to a complete ML workflow
implemented in PyTorch, with links to learn more about each of these concepts.
We’ll use the FashionMNIST dataset to train a neural network that predicts if an input image belongs
to one of the following classes: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker,
Bag, or Ankle boot.
This tutorial assumes a basic familiarity with Python and Deep Learning concepts.
Running the Tutorial Code
#
You can run this tutorial in a couple of ways:
In the cloud
: This is the easiest way to get started! Each section has a “Run in Google Colab” link at the top, which opens an integrated notebook in Google Colab with the code in a fully-hosted environment.
Locally
: This option requires you to set up PyTorch and TorchVision first on your local machine (
installation instructions
). Download the notebook or copy the code into your favorite IDE.
How to Use this Guide
#
If you’re familiar with other deep learning frameworks, check out the
0. Quickstart
first
to quickly familiarize yourself with PyTorch’s API.
If you’re new to deep learning frameworks, head right into the first section of our step-by-step guide:
1. Tensors
.
0.
Quickstart
1.
Tensors
2.
Datasets and DataLoaders
3.
Transforms
4.
Build Model
5.
Automatic Differentiation
6.
Optimization Loop
7.
Save, Load and Use Model
Download
Jupyter
notebook:
intro.ipynb
Download
Python
source
code:
intro.py
Download
zipped:
intro.zip
On this page
PyTorch Libraries
ExecuTorch
Helion
torchao
kineto
torchtitan
TorchRL
torchvision
torchaudio
tensordict
PyTorch on XLA Devices


---


## PyTorch Quickstart

**Source:** https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html

Rate this Page
★
★
★
★
★
beginner/basics/quickstart_tutorial
Run in Google Colab
Colab
Download Notebook
Notebook
View on GitHub
GitHub
Note
Go to the end
to download the full example code.
Learn the Basics
||
Quickstart
||
Tensors
||
Datasets & DataLoaders
||
Transforms
||
Build Model
||
Autograd
||
Optimization
||
Save & Load Model
Quickstart
#
Created On: Feb 09, 2021 | Last Updated: May 06, 2026 | Last Verified: Not Verified
This section runs through the API for common tasks in machine learning. Refer to the links in each section to dive deeper.
Working with data
#
PyTorch has two
primitives to work with data
:
torch.utils.data.DataLoader
and
torch.utils.data.Dataset
.
Dataset
stores the samples and their corresponding labels, and
DataLoader
wraps an iterable around
the
Dataset
.
import
torch
from
torch
import
nn
from
torch.utils.data
import
DataLoader
from
torchvision
import
datasets
from
torchvision.transforms
import
v2
PyTorch offers domain-specific libraries such as
TorchText
,
TorchVision
, and
TorchAudio
,
all of which include datasets. For this tutorial, we will be using a TorchVision dataset.
The
torchvision.datasets
module contains
Dataset
objects for many real-world vision data like
CIFAR, COCO (
full list here
). In this tutorial, we
use the FashionMNIST dataset. Every TorchVision
Dataset
includes two arguments:
transform
and
target_transform
to modify the samples and labels respectively.
# Download training data from open datasets.
training_data
=
datasets
.
FashionMNIST
(
root
=
"data"
,
train
=
True
,
download
=
True
,
transform
=
v2
.
Compose
([
v2
.
ToImage
(),
v2
.
ToDtype
(
torch
.
float32
,
scale
=
True
)]),
)
# Download test data from open datasets.
test_data
=
datasets
.
FashionMNIST
(
root
=
"data"
,
train
=
False
,
download
=
True
,
transform
=
v2
.
Compose
([
v2
.
ToImage
(),
v2
.
ToDtype
(
torch
.
float32
,
scale
=
True
)]),
)
0%| | 0.00/26.4M [00:00<?, ?B/s]
 0%| | 65.5k/26.4M [00:00<01:09, 377kB/s]
 1%| | 229k/26.4M [00:00<00:37, 708kB/s]
 3%|▎ | 918k/26.4M [00:00<00:11, 2.18MB/s]
 14%|█▍ | 3.67M/26.4M [00:00<00:03, 7.54MB/s]
 37%|███▋ | 9.70M/26.4M [00:00<00:00, 17.3MB/s]
 60%|█████▉ | 15.7M/26.4M [00:01<00:00, 23.1MB/s]
 82%|████████▏ | 21.7M/26.4M [00:01<00:00, 26.6MB/s]
100%|██████████| 26.4M/26.4M [00:01<00:00, 20.0MB/s]

 0%| | 0.00/29.5k [00:00<?, ?B/s]
100%|██████████| 29.5k/29.5k [00:00<00:00, 332kB/s]

 0%| | 0.00/4.42M [00:00<?, ?B/s]
 1%|▏ | 65.5k/4.42M [00:00<00:11, 366kB/s]
 5%|▌ | 229k/4.42M [00:00<00:06, 688kB/s]
 21%|██ | 918k/4.42M [00:00<00:01, 2.13MB/s]
 82%|████████▏ | 3.64M/4.42M [00:00<00:00, 8.10MB/s]
100%|██████████| 4.42M/4.42M [00:00<00:00, 6.15MB/s]

 0%| | 0.00/5.15k [00:00<?, ?B/s]
100%|██████████| 5.15k/5.15k [00:00<00:00, 51.0MB/s]
We pass the
Dataset
as an argument to
DataLoader
. This wraps an iterable over our dataset, and supports
automatic batching, sampling, shuffling and multiprocess data loading. Here we define a batch size of 64, i.e. each element
in the dataloader iterable will return a batch of 64 features and labels.
batch_size
=
64
# Create data loaders.
train_dataloader
=
DataLoader
(
training_data
,
batch_size
=
batch_size
)
test_dataloader
=
DataLoader
(
test_data
,
batch_size
=
batch_size
)
for
X
,
y
in
test_dataloader
:
print
(
f
"Shape of X [N, C, H, W]:
{
X
.
shape
}
"
)
print
(
f
"Shape of y:
{
y
.
shape
}
{
y
.
dtype
}
"
)
break
Shape of X [N, C, H, W]: torch.Size([64, 1, 28, 28])
Shape of y: torch.Size([64]) torch.int64
Read more about
loading data in PyTorch
.
Creating Models
#
To define a neural network in PyTorch, we create a class that inherits
from
nn.Module
. We define the layers of the network
in the
__init__
function and specify how data will pass through the network in the
forward
function. To accelerate
operations in the neural network, we move it to the
accelerator
such as CUDA, MPS, MTIA, or XPU. If the current accelerator is available, we will use it. Otherwise, we use the CPU.
device
=
torch
.
accelerator
.
current_accelerator
()
.
type
if
torch
.
accelerator
.
is_available
()
else
"cpu"
print
(
f
"Using
{
device
}
device"
)
# Define model
class
NeuralNetwork
(
nn
.
Module
):
def
__init__
(
self
):
super
()
.
__init__
()
self
.
flatten
=
nn
.
Flatten
()
self
.
linear_relu_stack
=
nn
.
Sequential
(
nn
.
Linear
(
28
*
28
,
512
),
nn
.
ReLU
(),
nn
.
Linear
(
512
,
512
),
nn
.
ReLU
(),
nn
.
Linear
(
512
,
10
)
)
def
forward
(
self
,
x
):
x
=
self
.
flatten
(
x
)
logits
=
self
.
linear_relu_stack
(
x
)
return
logits
model
=
NeuralNetwork
()
.
to
(
device
)
print
(
model
)
Using cuda device
NeuralNetwork(
 (flatten): Flatten(start_dim=1, end_dim=-1)
 (linear_relu_stack): Sequential(
 (0): Linear(in_features=784, out_features=512, bias=True)
 (1): ReLU()
 (2): Linear(in_features=512, out_features=512, bias=True)
 (3): ReLU()
 (4): Linear(in_features=512, out_features=10, bias=True)
 )
)
Read more about
building neural networks in PyTorch
.
Optimizing the Model Parameters
#
To train a model, we need a
loss function
and an
optimizer
.
loss_fn
=
nn
.
CrossEntropyLoss
()
optimizer
=
torch
.
optim
.
SGD
(
model
.
parameters
(),
lr
=
1e-3
)
In a single training loop, the model makes predictions on the training dataset (fed to it in batches), and
backpropagates the prediction error to adjust the model’s parameters.
def
train
(
dataloader
,
model
,
loss_fn
,
optimizer
):
size
=
len
(
dataloader
.
dataset
)
model
.
train
()
for
batch
,
(
X
,
y
)
in
enumerate
(
dataloader
):
X
,
y
=
X
.
to
(
device
),
y
.
to
(
device
)
# Compute prediction error
pred
=
model
(
X
)
loss
=
loss_fn
(
pred
,
y
)
# Backpropagation
loss
.
backward
()
optimizer
.
step
()
optimizer
.
zero_grad
()
if
batch
%
100
==
0
:
loss
,
current
=
loss
.
item
(),
(
batch
+
1
)
*
len
(
X
)
print
(
f
"loss:
{
loss
:
>7f
}
[
{
current
:
>5d
}
/
{
size
:
>5d
}
]"
)
We also check the model’s performance against the test dataset to ensure it is learning.
def
test
(
dataloader
,
model
,
loss_fn
):
size
=
len
(
dataloader
.
dataset
)
num_batches
=
len
(
dataloader
)
model
.
eval
()
test_loss
,
correct
=
0
,
0
with
torch
.
no_grad
():
for
X
,
y
in
dataloader
:
X
,
y
=
X
.
to
(
device
),
y
.
to
(
device
)
pred
=
model
(
X
)
test_loss
+=
loss_fn
(
pred
,
y
)
.
item
()
correct
+=
(
pred
.
argmax
(
1
)
==
y
)
.
type
(
torch
.
float
)
.
sum
()
.
item
()
test_loss
/=
num_batches
correct
/=
size
print
(
f
"Test Error:
\n
Accuracy:
{
(
100
*
correct
)
:
>0.1f
}
%, Avg loss:
{
test_loss
:
>8f
}
\n
"
)
The training process is conducted over several iterations (
epochs
). During each epoch, the model learns
parameters to make better predictions. We print the model’s accuracy and loss at each epoch; we’d like to see the
accuracy increase and the loss decrease with every epoch.
epochs
=
5
for
t
in
range
(
epochs
):
print
(
f
"Epoch
{
t
+
1
}
\n
-------------------------------"
)
train
(
train_dataloader
,
model
,
loss_fn
,
optimizer
)
test
(
test_dataloader
,
model
,
loss_fn
)
print
(
"Done!"
)
Epoch 1
-------------------------------
loss: 2.310647 [ 64/60000]
loss: 2.298127 [ 6464/60000]
loss: 2.280417 [12864/60000]
loss: 2.270391 [19264/60000]
loss: 2.258585 [25664/60000]
loss: 2.238101 [32064/60000]
loss: 2.234583 [38464/60000]
loss: 2.211815 [44864/60000]
loss: 2.205006 [51264/60000]
loss: 2.176353 [57664/60000]
Test Error:
 Accuracy: 53.3%, Avg loss: 2.169908

Epoch 2
-------------------------------
loss: 2.179659 [ 64/60000]
loss: 2.171518 [ 6464/60000]
loss: 2.121356 [12864/60000]
loss: 2.128750 [19264/60000]
loss: 2.092170 [25664/60000]
loss: 2.034117 [32064/60000]
loss: 2.054651 [38464/60000]
loss: 1.991964 [44864/60000]
loss: 1.987191 [51264/60000]
loss: 1.917593 [57664/60000]
Test Error:
 Accuracy: 56.1%, Avg loss: 1.918608

Epoch 3
-------------------------------
loss: 1.951120 [ 64/60000]
loss: 1.925469 [ 6464/60000]
loss: 1.820631 [12864/60000]
loss: 1.845260 [19264/60000]
loss: 1.753157 [25664/60000]
loss: 1.695571 [32064/60000]
loss: 1.707866 [38464/60000]
loss: 1.624048 [44864/60000]
loss: 1.633668 [51264/60000]
loss: 1.526472 [57664/60000]
Test Error:
 Accuracy: 59.0%, Avg loss: 1.547817

Epoch 4
-------------------------------
loss: 1.613080 [ 64/60000]
loss: 1.577875 [ 6464/60000]
loss: 1.437043 [12864/60000]
loss: 1.494159 [19264/60000]
loss: 1.389003 [25664/60000]
loss: 1.372263 [32064/60000]
loss: 1.381328 [38464/60000]
loss: 1.317837 [44864/60000]
loss: 1.338865 [51264/60000]
loss: 1.244815 [57664/60000]
Test Error:
 Accuracy: 62.9%, Avg loss: 1.270471

Epoch 5
-------------------------------
loss: 1.344652 [ 64/60000]
loss: 1.324580 [ 6464/60000]
loss: 1.169715 [12864/60000]
loss: 1.264619 [19264/60000]
loss: 1.152006 [25664/60000]
loss: 1.167499 [32064/60000]
loss: 1.186366 [38464/60000]
loss: 1.133232 [44864/60000]
loss: 1.157873 [51264/60000]
loss: 1.085890 [57664/60000]
Test Error:
 Accuracy: 64.8%, Avg loss: 1.102348

Done!
Read more about
Training your model
.
Saving Models
#
A common way to save a model is to serialize the internal state dictionary (containing the model parameters).
torch
.
save
(
model
.
state_dict
(),
"model.pth"
)
print
(
"Saved PyTorch Model State to model.pth"
)
Saved PyTorch Model State to model.pth
Loading Models
#
The process for loading a model includes re-creating the model structure and loading
the state dictionary into it.
model
=
NeuralNetwork
()
.
to
(
device
)
model
.
load_state_dict
(
torch
.
load
(
"model.pth"
,
weights_only
=
True
))
<All keys matched successfully>
This model can now be used to make predictions.
classes
=
[
"T-shirt/top"
,
"Trouser"
,
"Pullover"
,
"Dress"
,
"Coat"
,
"Sandal"
,
"Shirt"
,
"Sneaker"
,
"Bag"
,
"Ankle boot"
,
]
model
.
eval
()
x
,
y
=
test_data
[
0
][
0
],
test_data
[
0
][
1
]
with
torch
.
no_grad
():
x
=
x
.
to
(
device
)
pred
=
model
(
x
)
predicted
,
actual
=
classes
[
pred
[
0
]
.
argmax
(
0
)],
classes
[
y
]
print
(
f
'Predicted: "
{
predicted
}
", Actual: "
{
actual
}
"'
)
Predicted: "Ankle boot", Actual: "Ankle boot"
Read more about
Saving & Loading your model
.
Total running time of the script:
(0 minutes 56.038 seconds)
Download
Jupyter
notebook:
quickstart_tutorial.ipynb
Download
Python
source
code:
quickstart_tutorial.py
Download
zipped:
quickstart_tutorial.zip
On this page
PyTorch Libraries
ExecuTorch
Helion
torchao
kineto
torchtitan
TorchRL
torchvision
torchaudio
tensordict
PyTorch on XLA Devices


---


## PyTorch Tensors

**Source:** https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html

Rate this Page
★
★
★
★
★
beginner/basics/tensorqs_tutorial
Run in Google Colab
Colab
Download Notebook
Notebook
View on GitHub
GitHub
Note
Go to the end
to download the full example code.
Learn the Basics
||
Quickstart
||
Tensors
||
Datasets & DataLoaders
||
Transforms
||
Build Model
||
Autograd
||
Optimization
||
Save & Load Model
Tensors
#
Created On: Feb 10, 2021 | Last Updated: Apr 30, 2026 | Last Verified: Nov 05, 2024
Tensors are a specialized data structure that are very similar to arrays and matrices.
In PyTorch, we use tensors to encode the inputs and outputs of a model, as well as the model’s parameters.
Tensors are similar to
NumPy’s
ndarrays, except that tensors can run on GPUs or other hardware accelerators. In fact, tensors and
NumPy arrays can often share the same underlying memory, eliminating the need to copy data (see
Bridge with NumPy
). Tensors
are also optimized for automatic differentiation (we’ll see more about that later in the
Autograd
section). If you’re familiar with ndarrays, you’ll be right at home with the Tensor API. If not, follow along!
import
torch
import
numpy
as
np
Initializing a Tensor
#
Tensors can be initialized in various ways. Take a look at the following examples:
Directly from data
Tensors can be created directly from data. The data type is automatically inferred.
data
=
[[
1
,
2
],[
3
,
4
]]
x_data
=
torch
.
tensor
(
data
)
From a NumPy array
Tensors can be created from NumPy arrays (and vice versa - see
Bridge with NumPy
).
np_array
=
np
.
array
(
data
)
x_np
=
torch
.
from_numpy
(
np_array
)
From another tensor:
The new tensor retains the properties (shape, datatype) of the argument tensor, unless explicitly overridden.
x_ones
=
torch
.
ones_like
(
x_data
)
# retains the properties of x_data
print
(
f
"Ones Tensor:
\n
{
x_ones
}
\n
"
)
x_rand
=
torch
.
rand_like
(
x_data
,
dtype
=
torch
.
float
)
# overrides the datatype of x_data
print
(
f
"Random Tensor:
\n
{
x_rand
}
\n
"
)
Ones Tensor:
 tensor([[1, 1],
 [1, 1]])

Random Tensor:
 tensor([[0.6235, 0.2075],
 [0.6531, 0.8786]])
With random or constant values:
shape
is a tuple of tensor dimensions. In the functions below, it determines the dimensionality of the output tensor.
shape
=
(
2
,
3
)
rand_tensor
=
torch
.
rand
(
shape
)
ones_tensor
=
torch
.
ones
(
shape
)
zeros_tensor
=
torch
.
zeros
(
shape
)
print
(
f
"Random Tensor:
\n
{
rand_tensor
}
\n
"
)
print
(
f
"Ones Tensor:
\n
{
ones_tensor
}
\n
"
)
print
(
f
"Zeros Tensor:
\n
{
zeros_tensor
}
"
)
Random Tensor:
 tensor([[0.4521, 0.6433, 0.8520],
 [0.9149, 0.7588, 0.6311]])

Ones Tensor:
 tensor([[1., 1., 1.],
 [1., 1., 1.]])

Zeros Tensor:
 tensor([[0., 0., 0.],
 [0., 0., 0.]])
Attributes of a Tensor
#
Tensor attributes describe their shape, datatype, and the device on which they are stored.
tensor
=
torch
.
rand
(
3
,
4
)
print
(
f
"Shape of tensor:
{
tensor
.
shape
}
"
)
print
(
f
"Datatype of tensor:
{
tensor
.
dtype
}
"
)
print
(
f
"Device tensor is stored on:
{
tensor
.
device
}
"
)
Shape of tensor: torch.Size([3, 4])
Datatype of tensor: torch.float32
Device tensor is stored on: cpu
Operations on Tensors
#
Over 1200 tensor operations, including arithmetic, linear algebra, matrix manipulation (transposing,
indexing, slicing), sampling and more are
comprehensively described
here
.
Each of these operations can be run on the CPU and
Accelerator
such as CUDA, MPS, MTIA, or XPU. If you’re using Colab, allocate an accelerator by going to Runtime > Change runtime type > GPU.
By default, tensors are created on the CPU. We need to explicitly move tensors to the accelerator using
.to
method (after checking for accelerator availability). Keep in mind that copying large tensors
across devices can be expensive in terms of time and memory!
# We move our tensor to the current accelerator if available
if
torch
.
accelerator
.
is_available
():
tensor
=
tensor
.
to
(
torch
.
accelerator
.
current_accelerator
())
Try out some of the operations from the list.
If you’re familiar with the NumPy API, you’ll find the Tensor API a breeze to use.
Standard numpy-like indexing and slicing:
tensor
=
torch
.
ones
(
4
,
4
)
print
(
f
"First row:
{
tensor
[
0
]
}
"
)
print
(
f
"First column:
{
tensor
[:,
0
]
}
"
)
print
(
f
"Last column:
{
tensor
[
...
,
-
1
]
}
"
)
tensor
[:,
1
]
=
0
print
(
tensor
)
First row: tensor([1., 1., 1., 1.])
First column: tensor([1., 1., 1., 1.])
Last column: tensor([1., 1., 1., 1.])
tensor([[1., 0., 1., 1.],
 [1., 0., 1., 1.],
 [1., 0., 1., 1.],
 [1., 0., 1., 1.]])
Joining tensors
You can use
torch.cat
to concatenate a sequence of tensors along a given dimension.
See also
torch.stack
,
another tensor joining operator that is subtly different from
torch.cat
.
t1
=
torch
.
cat
([
tensor
,
tensor
,
tensor
],
dim
=
1
)
print
(
t1
)
tensor([[1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
 [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
 [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
 [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.]])
Arithmetic operations
# This computes the matrix multiplication between two tensors. y1, y2, y3 will have the same value
# ``tensor.T`` returns the transpose of a tensor
y1
=
tensor
@
tensor
.
T
y2
=
tensor
.
matmul
(
tensor
.
T
)
y3
=
torch
.
rand_like
(
y1
)
torch
.
matmul
(
tensor
,
tensor
.
T
,
out
=
y3
)
# This computes the element-wise product. z1, z2, z3 will have the same value
z1
=
tensor
*
tensor
z2
=
tensor
.
mul
(
tensor
)
z3
=
torch
.
rand_like
(
tensor
)
torch
.
mul
(
tensor
,
tensor
,
out
=
z3
)
tensor([[1., 0., 1., 1.],
 [1., 0., 1., 1.],
 [1., 0., 1., 1.],
 [1., 0., 1., 1.]])
Single-element tensors
If you have a one-element tensor, for example by aggregating all
values of a tensor into one value, you can convert it to a Python
numerical value using
item()
:
agg
=
tensor
.
sum
()
agg_item
=
agg
.
item
()
print
(
agg_item
,
type
(
agg_item
))
12.0 <class 'float'>
In-place operations
Operations that store the result into the operand are called in-place. They are denoted by a
_
suffix.
For example:
x.copy_(y)
,
x.t_()
, will change
x
.
print
(
f
"
{
tensor
}
\n
"
)
tensor
.
add_
(
5
)
print
(
tensor
)
tensor([[1., 0., 1., 1.],
 [1., 0., 1., 1.],
 [1., 0., 1., 1.],
 [1., 0., 1., 1.]])

tensor([[6., 5., 6., 6.],
 [6., 5., 6., 6.],
 [6., 5., 6., 6.],
 [6., 5., 6., 6.]])
Note
In-place operations save some memory, but can be problematic when computing derivatives because of an immediate loss
of history. Hence, their use is discouraged.
Bridge with NumPy
#
Tensors on the CPU and NumPy arrays can share their underlying memory
locations, and changing one will change the other.
Tensor to NumPy array
#
t
=
torch
.
ones
(
5
)
print
(
f
"t:
{
t
}
"
)
n
=
t
.
numpy
()
print
(
f
"n:
{
n
}
"
)
t: tensor([1., 1., 1., 1., 1.])
n: [1. 1. 1. 1. 1.]
A change in the tensor reflects in the NumPy array.
t
.
add_
(
1
)
print
(
f
"t:
{
t
}
"
)
print
(
f
"n:
{
n
}
"
)
t: tensor([2., 2., 2., 2., 2.])
n: [2. 2. 2. 2. 2.]
NumPy array to Tensor
#
n
=
np
.
ones
(
5
)
t
=
torch
.
from_numpy
(
n
)
Changes in the NumPy array reflects in the tensor.
np
.
add
(
n
,
1
,
out
=
n
)
print
(
f
"t:
{
t
}
"
)
print
(
f
"n:
{
n
}
"
)
t: tensor([2., 2., 2., 2., 2.], dtype=torch.float64)
n: [2. 2. 2. 2. 2.]
Total running time of the script:
(0 minutes 0.468 seconds)
Download
Jupyter
notebook:
tensorqs_tutorial.ipynb
Download
Python
source
code:
tensorqs_tutorial.py
Download
zipped:
tensorqs_tutorial.zip
On this page
PyTorch Libraries
ExecuTorch
Helion
torchao
kineto
torchtitan
TorchRL
torchvision
torchaudio
tensordict
PyTorch on XLA Devices


---


## Datasets and DataLoaders

**Source:** https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html

Rate this Page
★
★
★
★
★
beginner/basics/data_tutorial
Run in Google Colab
Colab
Download Notebook
Notebook
View on GitHub
GitHub
Note
Go to the end
to download the full example code.
Learn the Basics
||
Quickstart
||
Tensors
||
Datasets & DataLoaders
||
Transforms
||
Build Model
||
Autograd
||
Optimization
||
Save & Load Model
Datasets & DataLoaders
#
Created On: Feb 09, 2021 | Last Updated: May 07, 2026 | Last Verified: Nov 05, 2024
Code for processing data samples can get messy and hard to maintain; we ideally want our dataset code
to be decoupled from our model training code for better readability and modularity.
PyTorch provides two data primitives:
torch.utils.data.DataLoader
and
torch.utils.data.Dataset
that allow you to use pre-loaded datasets as well as your own data.
Dataset
stores the samples and their corresponding labels, and
DataLoader
wraps an iterable around
the
Dataset
to enable easy access to the samples.
PyTorch domain libraries provide a number of pre-loaded datasets (such as FashionMNIST) that
subclass
torch.utils.data.Dataset
and implement functions specific to the particular data.
They can be used to prototype and benchmark your model. You can find them
here:
Image Datasets
,
Text Datasets
, and
Audio Datasets
Loading a Dataset
#
Here is an example of how to load the
Fashion-MNIST
dataset from TorchVision.
Fashion-MNIST is a dataset of Zalando’s article images consisting of 60,000 training examples and 10,000 test examples.
Each example comprises a 28×28 grayscale image and an associated label from one of 10 classes.
We load the
FashionMNIST Dataset
with the following parameters:
root
is the path where the train/test data is stored,
train
specifies training or test dataset,
download=True
downloads the data from the internet if it’s not available at
root
.
transform
and
target_transform
specify the feature and label transformations
import
torch
from
torch.utils.data
import
Dataset
from
torchvision
import
datasets
from
torchvision.transforms
import
v2
import
matplotlib.pyplot
as
plt
training_data
=
datasets
.
FashionMNIST
(
root
=
"data"
,
train
=
True
,
download
=
True
,
transform
=
v2
.
Compose
([
v2
.
ToImage
(),
v2
.
ToDtype
(
torch
.
float32
,
scale
=
True
)])
)
test_data
=
datasets
.
FashionMNIST
(
root
=
"data"
,
train
=
False
,
download
=
True
,
transform
=
v2
.
Compose
([
v2
.
ToImage
(),
v2
.
ToDtype
(
torch
.
float32
,
scale
=
True
)])
)
0%| | 0.00/26.4M [00:00<?, ?B/s]
 0%| | 65.5k/26.4M [00:00<01:09, 377kB/s]
 1%| | 229k/26.4M [00:00<00:37, 706kB/s]
 3%|▎ | 918k/26.4M [00:00<00:11, 2.18MB/s]
 14%|█▍ | 3.67M/26.4M [00:00<00:03, 7.52MB/s]
 37%|███▋ | 9.73M/26.4M [00:00<00:00, 17.3MB/s]
 59%|█████▉ | 15.5M/26.4M [00:01<00:00, 22.7MB/s]
 81%|████████ | 21.4M/26.4M [00:01<00:00, 26.2MB/s]
100%|██████████| 26.4M/26.4M [00:01<00:00, 20.0MB/s]

 0%| | 0.00/29.5k [00:00<?, ?B/s]
100%|██████████| 29.5k/29.5k [00:00<00:00, 338kB/s]

 0%| | 0.00/4.42M [00:00<?, ?B/s]
 1%|▏ | 65.5k/4.42M [00:00<00:11, 373kB/s]
 5%|▌ | 229k/4.42M [00:00<00:05, 703kB/s]
 21%|██ | 918k/4.42M [00:00<00:01, 2.17MB/s]
 82%|████████▏ | 3.64M/4.42M [00:00<00:00, 8.17MB/s]
100%|██████████| 4.42M/4.42M [00:00<00:00, 6.28MB/s]

 0%| | 0.00/5.15k [00:00<?, ?B/s]
100%|██████████| 5.15k/5.15k [00:00<00:00, 43.7MB/s]
Iterating and Visualizing the Dataset
#
We can index
Datasets
manually like a list:
training_data[index]
.
We use
matplotlib
to visualize some samples in our training data.
labels_map
=
{
0
:
"T-Shirt"
,
1
:
"Trouser"
,
2
:
"Pullover"
,
3
:
"Dress"
,
4
:
"Coat"
,
5
:
"Sandal"
,
6
:
"Shirt"
,
7
:
"Sneaker"
,
8
:
"Bag"
,
9
:
"Ankle Boot"
,
}
figure
=
plt
.
figure
(
figsize
=
(
8
,
8
))
cols
,
rows
=
3
,
3
for
i
in
range
(
1
,
cols
*
rows
+
1
):
sample_idx
=
torch
.
randint
(
len
(
training_data
),
size
=
(
1
,))
.
item
()
img
,
label
=
training_data
[
sample_idx
]
figure
.
add_subplot
(
rows
,
cols
,
i
)
plt
.
title
(
labels_map
[
label
])
plt
.
axis
(
"off"
)
plt
.
imshow
(
img
.
squeeze
(),
cmap
=
"gray"
)
plt
.
show
()
Creating a Custom Dataset for your files
#
A custom Dataset class must implement three functions:
__init__
,
__len__
, and
__getitem__
.
Take a look at this implementation; the FashionMNIST images are stored
in a directory
img_dir
, and their labels are stored separately in a CSV file
annotations_file
.
In the next sections, we’ll break down what’s happening in each of these functions.
import
os
import
pandas
as
pd
from
torchvision.io
import
decode_image
class
CustomImageDataset
(
Dataset
):
def
__init__
(
self
,
annotations_file
,
img_dir
,
transform
=
None
,
target_transform
=
None
):
self
.
img_labels
=
pd
.
read_csv
(
annotations_file
)
self
.
img_dir
=
img_dir
self
.
transform
=
transform
self
.
target_transform
=
target_transform
def
__len__
(
self
):
return
len
(
self
.
img_labels
)
def
__getitem__
(
self
,
idx
):
img_path
=
os
.
path
.
join
(
self
.
img_dir
,
self
.
img_labels
.
iloc
[
idx
,
0
])
image
=
decode_image
(
img_path
)
label
=
self
.
img_labels
.
iloc
[
idx
,
1
]
if
self
.
transform
:
image
=
self
.
transform
(
image
)
if
self
.
target_transform
:
label
=
self
.
target_transform
(
label
)
return
image
,
label
__init__
#
The __init__ function is run once when instantiating the Dataset object. We initialize
the directory containing the images, the annotations file, and both transforms (covered
in more detail in the next section).
The labels.csv file looks like:
tshirt1
.
jpg
,
0
tshirt2
.
jpg
,
0
......
ankleboot999
.
jpg
,
9
def
__init__
(
self
,
annotations_file
,
img_dir
,
transform
=
None
,
target_transform
=
None
):
self
.
img_labels
=
pd
.
read_csv
(
annotations_file
)
self
.
img_dir
=
img_dir
self
.
transform
=
transform
self
.
target_transform
=
target_transform
__len__
#
The __len__ function returns the number of samples in our dataset.
Example:
def
__len__
(
self
):
return
len
(
self
.
img_labels
)
__getitem__
#
The __getitem__ function loads and returns a sample from the dataset at the given index
idx
.
Based on the index, it identifies the image’s location on disk, converts that to a tensor using
decode_image
, retrieves the
corresponding label from the csv data in
self.img_labels
, calls the transform functions on them (if applicable), and returns the
tensor image and corresponding label in a tuple.
def
__getitem__
(
self
,
idx
):
img_path
=
os
.
path
.
join
(
self
.
img_dir
,
self
.
img_labels
.
iloc
[
idx
,
0
])
image
=
decode_image
(
img_path
)
label
=
self
.
img_labels
.
iloc
[
idx
,
1
]
if
self
.
transform
:
image
=
self
.
transform
(
image
)
if
self
.
target_transform
:
label
=
self
.
target_transform
(
label
)
return
image
,
label
Preparing your data for training with DataLoaders
#
The
Dataset
retrieves our dataset’s features and labels one sample at a time. While training a model, we typically want to
pass samples in “minibatches”, reshuffle the data at every epoch to reduce model overfitting, and use Python’s
multiprocessing
to
speed up data retrieval.
DataLoader
is an iterable that abstracts this complexity for us in an easy API.
from
torch.utils.data
import
DataLoader
train_dataloader
=
DataLoader
(
training_data
,
batch_size
=
64
,
shuffle
=
True
)
test_dataloader
=
DataLoader
(
test_data
,
batch_size
=
64
,
shuffle
=
True
)
Iterate through the DataLoader
#
We have loaded that dataset into the
DataLoader
and can iterate through the dataset as needed.
Each iteration below returns a batch of
train_features
and
train_labels
(containing
batch_size=64
features and labels respectively).
Because we specified
shuffle=True
, after we iterate over all batches the data is shuffled (for finer-grained control over
the data loading order, take a look at
Samplers
).
# Display image and label.
train_features
,
train_labels
=
next
(
iter
(
train_dataloader
))
print
(
f
"Feature batch shape:
{
train_features
.
size
()
}
"
)
print
(
f
"Labels batch shape:
{
train_labels
.
size
()
}
"
)
img
=
train_features
[
0
]
.
squeeze
()
label
=
train_labels
[
0
]
plt
.
imshow
(
img
,
cmap
=
"gray"
)
plt
.
show
()
print
(
f
"Label:
{
label
}
"
)
Feature batch shape: torch.Size([64, 1, 28, 28])
Labels batch shape: torch.Size([64])
Label: 0
Further Reading
#
torch.utils.data API
Total running time of the script:
(0 minutes 4.863 seconds)
Download
Jupyter
notebook:
data_tutorial.ipynb
Download
Python
source
code:
data_tutorial.py
Download
zipped:
data_tutorial.zip
On this page
PyTorch Libraries
ExecuTorch
Helion
torchao
kineto
torchtitan
TorchRL
torchvision
torchaudio
tensordict
PyTorch on XLA Devices


---


## Building the Neural Network

**Source:** https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html

Rate this Page
★
★
★
★
★
beginner/basics/buildmodel_tutorial
Run in Google Colab
Colab
Download Notebook
Notebook
View on GitHub
GitHub
Note
Go to the end
to download the full example code.
Learn the Basics
||
Quickstart
||
Tensors
||
Datasets & DataLoaders
||
Transforms
||
Build Model
||
Autograd
||
Optimization
||
Save & Load Model
Build the Neural Network
#
Created On: Feb 09, 2021 | Last Updated: Aug 25, 2026 | Last Verified: Not Verified
Neural networks comprise of layers/modules that perform operations on data.
The
torch.nn
namespace provides all the building blocks you need to
build your own neural network. Every module in PyTorch subclasses the
nn.Module
.
A neural network is a module itself that consists of other modules (layers). This nested structure allows for
building and managing complex architectures easily.
In the following sections, we’ll build a neural network to classify images in the FashionMNIST dataset.
import
os
import
torch
from
torch
import
nn
from
torch.utils.data
import
DataLoader
from
torchvision
import
datasets
,
transforms
Get Device for Training
#
We want to be able to train our model on an
accelerator
such as CUDA, MPS, MTIA, or XPU. If the current accelerator is available, we will use it. Otherwise, we use the CPU.
device
=
torch
.
accelerator
.
current_accelerator
()
.
type
if
torch
.
accelerator
.
is_available
()
else
"cpu"
print
(
f
"Using
{
device
}
device"
)
Using cuda device
Define the Class
#
We define our neural network by subclassing
nn.Module
, and
initialize the neural network layers in
__init__
. Every
nn.Module
subclass implements
the operations on input data in the
forward
method.
class
NeuralNetwork
(
nn
.
Module
):
def
__init__
(
self
):
super
()
.
__init__
()
self
.
flatten
=
nn
.
Flatten
()
self
.
linear_relu_stack
=
nn
.
Sequential
(
nn
.
Linear
(
28
*
28
,
512
),
nn
.
ReLU
(),
nn
.
Linear
(
512
,
512
),
nn
.
ReLU
(),
nn
.
Linear
(
512
,
10
),
)
def
forward
(
self
,
x
):
x
=
self
.
flatten
(
x
)
logits
=
self
.
linear_relu_stack
(
x
)
return
logits
We create an instance of
NeuralNetwork
, and move it to the
device
, and print
its structure.
model
=
NeuralNetwork
()
.
to
(
device
)
print
(
model
)
NeuralNetwork(
 (flatten): Flatten(start_dim=1, end_dim=-1)
 (linear_relu_stack): Sequential(
 (0): Linear(in_features=784, out_features=512, bias=True)
 (1): ReLU()
 (2): Linear(in_features=512, out_features=512, bias=True)
 (3): ReLU()
 (4): Linear(in_features=512, out_features=10, bias=True)
 )
)
To use the model, we pass it the input data. This executes the model’s
forward
,
along with some
background operations
.
Do not call
model.forward()
directly!
Calling the model on the input returns a 2-dimensional tensor with dim=0 corresponding to each output of 10 raw predicted values for each class, and dim=1 corresponding to the individual values of each output.
We get the prediction probabilities by passing it through an instance of the
nn.Softmax
module.
X
=
torch
.
rand
(
1
,
28
,
28
,
device
=
device
)
logits
=
model
(
X
)
pred_probab
=
nn
.
Softmax
(
dim
=
1
)(
logits
)
y_pred
=
pred_probab
.
argmax
(
1
)
print
(
f
"Predicted class:
{
y_pred
}
"
)
Predicted class: tensor([3], device='cuda:0')
Model Layers
#
Let’s break down the layers in the FashionMNIST model. To illustrate it, we
will take a sample minibatch of 3 images of size 28x28 and see what happens to it as
we pass it through the network.
input_image
=
torch
.
rand
(
3
,
28
,
28
)
print
(
input_image
.
size
())
torch.Size([3, 28, 28])
nn.Flatten
#
We initialize the
nn.Flatten
layer to convert each 2D 28x28 image into a contiguous array of 784 pixel values (
the minibatch dimension (at dim=0) is maintained).
flatten
=
nn
.
Flatten
()
flat_image
=
flatten
(
input_image
)
print
(
flat_image
.
size
())
torch.Size([3, 784])
nn.Linear
#
The
linear layer
is a module that applies a linear transformation on the input using its stored weights and biases.
layer1
=
nn
.
Linear
(
in_features
=
28
*
28
,
out_features
=
20
)
hidden1
=
layer1
(
flat_image
)
print
(
hidden1
.
size
())
torch.Size([3, 20])
nn.ReLU
#
Non-linear activations are what create the complex mappings between the model’s inputs and outputs.
They are applied after linear transformations to introduce
nonlinearity
, helping neural networks
learn a wide variety of phenomena.
In this model, we use
nn.ReLU
between our
linear layers, but there’s other activations to introduce non-linearity in your model.
print
(
f
"Before ReLU:
{
hidden1
}
\n\n
"
)
hidden1
=
nn
.
ReLU
()(
hidden1
)
print
(
f
"After ReLU:
{
hidden1
}
"
)
Before ReLU: tensor([[ 0.3144, 0.2564, 0.0715, 0.0674, 0.0325, -0.2803, -0.2889, 0.3323,
 0.2607, -0.2992, 0.2304, 0.0465, 0.4094, -0.2759, -0.0787, 0.2125,
 0.3665, 0.3805, 0.1715, -0.1769],
 [ 0.3755, 0.2323, 0.4540, 0.2946, -0.0761, -0.2886, -0.4866, 0.3108,
 0.0549, -0.3030, 0.4530, -0.1943, 0.0159, 0.1030, 0.1582, 0.1848,
 0.5976, 0.4473, 0.5021, -0.3442],
 [-0.0374, 0.3474, 0.0851, 0.1591, -0.2234, 0.0957, -0.4159, 0.0961,
 -0.0009, -0.3636, 0.4018, 0.0325, 0.4114, 0.0045, -0.2395, -0.1824,
 0.5798, 0.3353, 0.5531, -0.3168]], grad_fn=<AddmmBackward0>)

After ReLU: tensor([[0.3144, 0.2564, 0.0715, 0.0674, 0.0325, 0.0000, 0.0000, 0.3323, 0.2607,
 0.0000, 0.2304, 0.0465, 0.4094, 0.0000, 0.0000, 0.2125, 0.3665, 0.3805,
 0.1715, 0.0000],
 [0.3755, 0.2323, 0.4540, 0.2946, 0.0000, 0.0000, 0.0000, 0.3108, 0.0549,
 0.0000, 0.4530, 0.0000, 0.0159, 0.1030, 0.1582, 0.1848, 0.5976, 0.4473,
 0.5021, 0.0000],
 [0.0000, 0.3474, 0.0851, 0.1591, 0.0000, 0.0957, 0.0000, 0.0961, 0.0000,
 0.0000, 0.4018, 0.0325, 0.4114, 0.0045, 0.0000, 0.0000, 0.5798, 0.3353,
 0.5531, 0.0000]], grad_fn=<ReluBackward0>)
nn.Sequential
#
nn.Sequential
is an ordered
container of modules. The data is passed through all the modules in the same order as defined. You can use
sequential containers to put together a quick network like
seq_modules
.
seq_modules
=
nn
.
Sequential
(
flatten
,
layer1
,
nn
.
ReLU
(),
nn
.
Linear
(
20
,
10
)
)
input_image
=
torch
.
rand
(
3
,
28
,
28
)
logits
=
seq_modules
(
input_image
)
nn.Softmax
#
The last linear layer of the neural network returns
logits
- raw values in [-infty, infty] - which are passed to the
nn.Softmax
module. The logits are scaled to values
[0, 1] representing the model’s predicted probabilities for each class.
dim
parameter indicates the dimension along
which the values must sum to 1.
softmax
=
nn
.
Softmax
(
dim
=
1
)
pred_probab
=
softmax
(
logits
)
Model Parameters
#
Many layers inside a neural network are
parameterized
, i.e. have associated weights
and biases that are optimized during training. Subclassing
nn.Module
automatically
tracks all fields defined inside your model object, and makes all parameters
accessible using your model’s
parameters()
or
named_parameters()
methods.
In this example, we iterate over each parameter, and print its size and a preview of its values.
print
(
f
"Model structure:
{
model
}
\n\n
"
)
for
name
,
param
in
model
.
named_parameters
():
print
(
f
"Layer:
{
name
}
| Size:
{
param
.
size
()
}
| Values :
{
param
[:
2
]
}
\n
"
)
Model structure: NeuralNetwork(
 (flatten): Flatten(start_dim=1, end_dim=-1)
 (linear_relu_stack): Sequential(
 (0): Linear(in_features=784, out_features=512, bias=True)
 (1): ReLU()
 (2): Linear(in_features=512, out_features=512, bias=True)
 (3): ReLU()
 (4): Linear(in_features=512, out_features=10, bias=True)
 )
)

Layer: linear_relu_stack.0.weight | Size: torch.Size([512, 784]) | Values : tensor([[-0.0166, -0.0180, 0.0229, ..., -0.0333, 0.0246, 0.0229],
 [ 0.0009, 0.0007, -0.0021, ..., -0.0105, 0.0037, 0.0194]],
 device='cuda:0', grad_fn=<SliceBackward0>)

Layer: linear_relu_stack.0.bias | Size: torch.Size([512]) | Values : tensor([-0.0041, 0.0117], device='cuda:0', grad_fn=<SliceBackward0>)

Layer: linear_relu_stack.2.weight | Size: torch.Size([512, 512]) | Values : tensor([[-0.0193, 0.0051, 0.0071, ..., -0.0127, -0.0063, 0.0082],
 [-0.0305, 0.0187, 0.0331, ..., -0.0260, -0.0273, -0.0340]],
 device='cuda:0', grad_fn=<SliceBackward0>)

Layer: linear_relu_stack.2.bias | Size: torch.Size([512]) | Values : tensor([0.0338, 0.0209], device='cuda:0', grad_fn=<SliceBackward0>)

Layer: linear_relu_stack.4.weight | Size: torch.Size([10, 512]) | Values : tensor([[ 0.0138, -0.0070, 0.0328, ..., -0.0009, -0.0019, -0.0256],
 [ 0.0058, 0.0398, 0.0408, ..., -0.0305, 0.0255, 0.0006]],
 device='cuda:0', grad_fn=<SliceBackward0>)

Layer: linear_relu_stack.4.bias | Size: torch.Size([10]) | Values : tensor([-0.0280, -0.0051], device='cuda:0', grad_fn=<SliceBackward0>)
Further Reading
#
torch.nn API
Total running time of the script:
(0 minutes 0.497 seconds)
Download
Jupyter
notebook:
buildmodel_tutorial.ipynb
Download
Python
source
code:
buildmodel_tutorial.py
Download
zipped:
buildmodel_tutorial.zip
On this page
PyTorch Libraries
ExecuTorch
Helion
torchao
kineto
torchtitan
TorchRL
torchvision
torchaudio
tensordict
PyTorch on XLA Devices


---


## Automatic Differentiation

**Source:** https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html

Rate this Page
★
★
★
★
★
torch.autograd
">
beginner/basics/autogradqs_tutorial
Run in Google Colab
Colab
Download Notebook
Notebook
View on GitHub
GitHub
Note
Go to the end
to download the full example code.
Learn the Basics
||
Quickstart
||
Tensors
||
Datasets & DataLoaders
||
Transforms
||
Build Model
||
Autograd
||
Optimization
||
Save & Load Model
Automatic Differentiation with
torch.autograd
#
Created On: Feb 10, 2021 | Last Updated: Jan 16, 2024 | Last Verified: Nov 05, 2024
When training neural networks, the most frequently used algorithm is
back propagation
. In this algorithm, parameters (model weights) are
adjusted according to the
gradient
of the loss function with respect
to the given parameter.
To compute those gradients, PyTorch has a built-in differentiation engine
called
torch.autograd
. It supports automatic computation of gradient for any
computational graph.
Consider the simplest one-layer neural network, with input
x
,
parameters
w
and
b
, and some loss function. It can be defined in
PyTorch in the following manner:
import
torch
x
=
torch
.
ones
(
5
)
# input tensor
y
=
torch
.
zeros
(
3
)
# expected output
w
=
torch
.
randn
(
5
,
3
,
requires_grad
=
True
)
b
=
torch
.
randn
(
3
,
requires_grad
=
True
)
z
=
torch
.
matmul
(
x
,
w
)
+
b
loss
=
torch
.
nn
.
functional
.
binary_cross_entropy_with_logits
(
z
,
y
)
Tensors, Functions and Computational graph
#
This code defines the following
computational graph
:
In this network,
w
and
b
are
parameters
, which we need to
optimize. Thus, we need to be able to compute the gradients of loss
function with respect to those variables. In order to do that, we set
the
requires_grad
property of those tensors.
Note
You can set the value of
requires_grad
when creating a
tensor, or later by using
x.requires_grad_(True)
method.
A function that we apply to tensors to construct computational graph is
in fact an object of class
Function
. This object knows how to
compute the function in the
forward
direction, and also how to compute
its derivative during the
backward propagation
step. A reference to
the backward propagation function is stored in
grad_fn
property of a
tensor. You can find more information of
Function
in the
documentation
.
print
(
f
"Gradient function for z =
{
z
.
grad_fn
}
"
)
print
(
f
"Gradient function for loss =
{
loss
.
grad_fn
}
"
)
Gradient function for z = <AddBackward0 object at 0x7f9b346651e0>
Gradient function for loss = <BinaryCrossEntropyWithLogitsBackward0 object at 0x7f9b34666980>
Computing Gradients
#
To optimize weights of parameters in the neural network, we need to
compute the derivatives of our loss function with respect to parameters,
namely, we need
\(\frac{\partial loss}{\partial w}\)
and
\(\frac{\partial loss}{\partial b}\)
under some fixed values of
x
and
y
. To compute those derivatives, we call
loss.backward()
, and then retrieve the values from
w.grad
and
b.grad
:
loss
.
backward
()
print
(
w
.
grad
)
print
(
b
.
grad
)
tensor([[0.1700, 0.1906, 0.0093],
 [0.1700, 0.1906, 0.0093],
 [0.1700, 0.1906, 0.0093],
 [0.1700, 0.1906, 0.0093],
 [0.1700, 0.1906, 0.0093]])
tensor([0.1700, 0.1906, 0.0093])
Note
We can only obtain the
grad
properties for the leaf
nodes of the computational graph, which have
requires_grad
property
set to
True
. For all other nodes in our graph, gradients will not be
available.
We can only perform gradient calculations using
backward
once on a given graph, for performance reasons. If we need
to do several
backward
calls on the same graph, we need to pass
retain_graph=True
to the
backward
call.
Disabling Gradient Tracking
#
By default, all tensors with
requires_grad=True
are tracking their
computational history and support gradient computation. However, there
are some cases when we do not need to do that, for example, when we have
trained the model and just want to apply it to some input data, i.e. we
only want to do
forward
computations through the network. We can stop
tracking computations by surrounding our computation code with
torch.no_grad()
block:
z
=
torch
.
matmul
(
x
,
w
)
+
b
print
(
z
.
requires_grad
)
with
torch
.
no_grad
():
z
=
torch
.
matmul
(
x
,
w
)
+
b
print
(
z
.
requires_grad
)
True
False
Another way to achieve the same result is to use the
detach()
method
on the tensor:
z
=
torch
.
matmul
(
x
,
w
)
+
b
z_det
=
z
.
detach
()
print
(
z_det
.
requires_grad
)
False
There are reasons you might want to disable gradient tracking:
To mark some parameters in your neural network as
frozen parameters
.
To
speed up computations
when you are only doing forward pass, because computations on tensors that do
not track gradients would be more efficient.
More on Computational Graphs
#
Conceptually, autograd keeps a record of data (tensors) and all executed
operations (along with the resulting new tensors) in a directed acyclic
graph (DAG) consisting of
Function
objects. In this DAG, leaves are the input tensors, roots are the output
tensors. By tracing this graph from roots to leaves, you can
automatically compute the gradients using the chain rule.
In a forward pass, autograd does two things simultaneously:
run the requested operation to compute a resulting tensor
maintain the operation’s
gradient function
in the DAG.
The backward pass kicks off when
.backward()
is called on the DAG
root.
autograd
then:
computes the gradients from each
.grad_fn
,
accumulates them in the respective tensor’s
.grad
attribute
using the chain rule, propagates all the way to the leaf tensors.
Note
DAGs are dynamic in PyTorch
An important thing to note is that the graph is recreated from scratch; after each
.backward()
call, autograd starts populating a new graph. This is
exactly what allows you to use control flow statements in your model;
you can change the shape, size and operations at every iteration if
needed.
Optional Reading: Tensor Gradients and Jacobian Products
#
In many cases, we have a scalar loss function, and we need to compute
the gradient with respect to some parameters. However, there are cases
when the output function is an arbitrary tensor. In this case, PyTorch
allows you to compute so-called
Jacobian product
, and not the actual
gradient.
For a vector function
\(\vec{y}=f(\vec{x})\)
, where
\(\vec{x}=\langle x_1,\dots,x_n\rangle\)
and
\(\vec{y}=\langle y_1,\dots,y_m\rangle\)
, a gradient of
\(\vec{y}\)
with respect to
\(\vec{x}\)
is given by
Jacobian
matrix
:
\[J=\left(\begin{array}{ccc}
 \frac{\partial y_{1}}{\partial x_{1}} & \cdots & \frac{\partial y_{1}}{\partial x_{n}}\\
 \vdots & \ddots & \vdots\\
 \frac{\partial y_{m}}{\partial x_{1}} & \cdots & \frac{\partial y_{m}}{\partial x_{n}}
 \end{array}\right)\]
Instead of computing the Jacobian matrix itself, PyTorch allows you to
compute
Jacobian Product
\(v^T\cdot J\)
for a given input vector
\(v=(v_1 \dots v_m)\)
. This is achieved by calling
backward
with
\(v\)
as an argument. The size of
\(v\)
should be the same as
the size of the original tensor, with respect to which we want to
compute the product:
inp
=
torch
.
eye
(
4
,
5
,
requires_grad
=
True
)
out
=
(
inp
+
1
)
.
pow
(
2
)
.
t
()
out
.
backward
(
torch
.
ones_like
(
out
),
retain_graph
=
True
)
print
(
f
"First call
\n
{
inp
.
grad
}
"
)
out
.
backward
(
torch
.
ones_like
(
out
),
retain_graph
=
True
)
print
(
f
"
\n
Second call
\n
{
inp
.
grad
}
"
)
inp
.
grad
.
zero_
()
out
.
backward
(
torch
.
ones_like
(
out
),
retain_graph
=
True
)
print
(
f
"
\n
Call after zeroing gradients
\n
{
inp
.
grad
}
"
)
First call
tensor([[4., 2., 2., 2., 2.],
 [2., 4., 2., 2., 2.],
 [2., 2., 4., 2., 2.],
 [2., 2., 2., 4., 2.]])

Second call
tensor([[8., 4., 4., 4., 4.],
 [4., 8., 4., 4., 4.],
 [4., 4., 8., 4., 4.],
 [4., 4., 4., 8., 4.]])

Call after zeroing gradients
tensor([[4., 2., 2., 2., 2.],
 [2., 4., 2., 2., 2.],
 [2., 2., 4., 2., 2.],
 [2., 2., 2., 4., 2.]])
Notice that when we call
backward
for the second time with the same
argument, the value of the gradient is different. This happens because
when doing
backward
propagation, PyTorch
accumulates the
gradients
, i.e. the value of computed gradients is added to the
grad
property of all leaf nodes of computational graph. If you want
to compute the proper gradients, you need to zero out the
grad
property before. In real-life training an
optimizer
helps us to do
this.
Note
Previously we were calling
backward()
function without
parameters. This is essentially equivalent to calling
backward(torch.tensor(1.0))
, which is a useful way to compute the
gradients in case of a scalar-valued function, such as loss during
neural network training.
Further Reading
#
Autograd Mechanics
Total running time of the script:
(0 minutes 0.359 seconds)
Download
Jupyter
notebook:
autogradqs_tutorial.ipynb
Download
Python
source
code:
autogradqs_tutorial.py
Download
zipped:
autogradqs_tutorial.zip
On this page
PyTorch Libraries
ExecuTorch
Helion
torchao
kineto
torchtitan
TorchRL
torchvision
torchaudio
tensordict
PyTorch on XLA Devices


---


## Optimization

**Source:** https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html

Rate this Page
★
★
★
★
★
beginner/basics/optimization_tutorial
Run in Google Colab
Colab
Download Notebook
Notebook
View on GitHub
GitHub
Note
Go to the end
to download the full example code.
Learn the Basics
||
Quickstart
||
Tensors
||
Datasets & DataLoaders
||
Transforms
||
Build Model
||
Autograd
||
Optimization
||
Save & Load Model
Optimizing Model Parameters
#
Created On: Feb 09, 2021 | Last Updated: Jul 31, 2026 | Last Verified: Nov 05, 2024
Now that we have a model and data it’s time to train, validate and test our model by optimizing its parameters on
our data. Training a model is an iterative process; in each iteration the model makes a guess about the output, calculates
the error in its guess (
loss
), collects the derivatives of the error with respect to its parameters (as we saw in
the
previous section
), and
optimizes
these parameters using gradient descent. For a more
detailed walkthrough of this process, check out this video on
backpropagation from 3Blue1Brown
.
Prerequisite Code
#
We load the code from the previous sections on
Datasets & DataLoaders
and
Build Model
.
import
torch
from
torch
import
nn
from
torch.utils.data
import
DataLoader
from
torchvision
import
datasets
from
torchvision.transforms
import
v2
training_data
=
datasets
.
FashionMNIST
(
root
=
"data"
,
train
=
True
,
download
=
True
,
transform
=
v2
.
Compose
([
v2
.
ToImage
(),
v2
.
ToDtype
(
torch
.
float32
,
scale
=
True
)])
)
test_data
=
datasets
.
FashionMNIST
(
root
=
"data"
,
train
=
False
,
download
=
True
,
transform
=
v2
.
Compose
([
v2
.
ToImage
(),
v2
.
ToDtype
(
torch
.
float32
,
scale
=
True
)])
)
train_dataloader
=
DataLoader
(
training_data
,
batch_size
=
64
)
test_dataloader
=
DataLoader
(
test_data
,
batch_size
=
64
)
class
NeuralNetwork
(
nn
.
Module
):
def
__init__
(
self
):
super
()
.
__init__
()
self
.
flatten
=
nn
.
Flatten
()
self
.
linear_relu_stack
=
nn
.
Sequential
(
nn
.
Linear
(
28
*
28
,
512
),
nn
.
ReLU
(),
nn
.
Linear
(
512
,
512
),
nn
.
ReLU
(),
nn
.
Linear
(
512
,
10
),
)
def
forward
(
self
,
x
):
x
=
self
.
flatten
(
x
)
logits
=
self
.
linear_relu_stack
(
x
)
return
logits
model
=
NeuralNetwork
()
0%| | 0.00/26.4M [00:00<?, ?B/s]
 0%| | 65.5k/26.4M [00:00<01:11, 366kB/s]
 1%| | 229k/26.4M [00:00<00:38, 686kB/s]
 3%|▎ | 918k/26.4M [00:00<00:12, 2.11MB/s]
 14%|█▍ | 3.67M/26.4M [00:00<00:03, 7.30MB/s]
 36%|███▌ | 9.54M/26.4M [00:00<00:00, 17.6MB/s]
 48%|████▊ | 12.8M/26.4M [00:00<00:00, 19.9MB/s]
 71%|███████ | 18.7M/26.4M [00:01<00:00, 26.4MB/s]
 84%|████████▎ | 22.1M/26.4M [00:01<00:00, 26.6MB/s]
100%|██████████| 26.4M/26.4M [00:01<00:00, 19.4MB/s]

 0%| | 0.00/29.5k [00:00<?, ?B/s]
100%|██████████| 29.5k/29.5k [00:00<00:00, 335kB/s]

 0%| | 0.00/4.42M [00:00<?, ?B/s]
 1%|▏ | 65.5k/4.42M [00:00<00:11, 366kB/s]
 4%|▎ | 164k/4.42M [00:00<00:08, 474kB/s]
 16%|█▌ | 688k/4.42M [00:00<00:02, 1.59MB/s]
 62%|██████▏ | 2.75M/4.42M [00:00<00:00, 5.52MB/s]
100%|██████████| 4.42M/4.42M [00:00<00:00, 6.15MB/s]

 0%| | 0.00/5.15k [00:00<?, ?B/s]
100%|██████████| 5.15k/5.15k [00:00<00:00, 58.5MB/s]
Hyperparameters
#
Hyperparameters are adjustable parameters that let you control the model optimization process.
Different hyperparameter values can impact model training and convergence rates
(
read more
about hyperparameter tuning)
We define the following hyperparameters for training:
Number of Epochs
- the number of times to iterate over the dataset
Batch Size
- the number of data samples propagated through the network before the parameters are updated
Learning Rate
- how much to update models parameters at each batch/epoch. Smaller values yield slow learning speed, while large values may result in unpredictable behavior during training.
learning_rate
=
1e-3
batch_size
=
64
epochs
=
5
Optimization Loop
#
Once we set our hyperparameters, we can then train and optimize our model with an optimization loop. Each
iteration of the optimization loop is called an
epoch
.
Each epoch consists of two main parts:
The Train Loop
- iterate over the training dataset and try to converge to optimal parameters.
The Validation/Test Loop
- iterate over the test dataset to check if model performance is improving.
Let’s briefly familiarize ourselves with some of the concepts used in the training loop. Jump ahead to
see the
Full Implementation
of the optimization loop.
Loss Function
#
When presented with some training data, our untrained network is likely not to give the correct
answer.
Loss function
measures the degree of dissimilarity of obtained result to the target value,
and it is the loss function that we want to minimize during training. To calculate the loss we make a
prediction using the inputs of our given data sample and compare it against the true data label value.
Common loss functions include
nn.MSELoss
(Mean Square Error) for regression tasks, and
nn.NLLLoss
(Negative Log Likelihood) for classification.
nn.CrossEntropyLoss
combines
nn.LogSoftmax
and
nn.NLLLoss
.
We pass our model’s output logits to
nn.CrossEntropyLoss
, which will normalize the logits and compute the prediction error.
# Initialize the loss function
loss_fn
=
nn
.
CrossEntropyLoss
()
Optimizer
#
Optimization is the process of adjusting model parameters to reduce model error in each training step.
Optimization algorithms
define how this process is performed (in this example we use Stochastic Gradient Descent).
All optimization logic is encapsulated in the
optimizer
object. Here, we use the SGD optimizer; additionally, there are many
different optimizers
available in PyTorch such as ADAM and RMSProp, that work better for different kinds of models and data.
We initialize the optimizer by registering the model’s parameters that need to be trained, and passing in the learning rate hyperparameter.
optimizer
=
torch
.
optim
.
SGD
(
model
.
parameters
(),
lr
=
learning_rate
)
Inside the training loop, optimization happens in three steps:
Call
optimizer.zero_grad()
to reset the gradients of model parameters. Gradients by default add up; to prevent double-counting, we explicitly zero them at each iteration.
Backpropagate the prediction loss with a call to
loss.backward()
. PyTorch deposits the gradients of the loss w.r.t. each parameter.
Once we have our gradients, we call
optimizer.step()
to adjust the parameters by the gradients collected in the backward pass.
Full Implementation
#
We define
train_loop
that loops over our optimization code, and
test_loop
that
evaluates the model’s performance against our test data.
def
train_loop
(
dataloader
,
model
,
loss_fn
,
optimizer
):
size
=
len
(
dataloader
.
dataset
)
# Set the model to training mode - important for batch normalization and dropout layers
# Unnecessary in this situation but added for best practices
model
.
train
()
for
batch
,
(
X
,
y
)
in
enumerate
(
dataloader
):
# Compute prediction and loss
pred
=
model
(
X
)
loss
=
loss_fn
(
pred
,
y
)
# Backpropagation
loss
.
backward
()
optimizer
.
step
()
optimizer
.
zero_grad
()
if
batch
%
100
==
0
:
loss
,
current
=
loss
.
item
(),
batch
*
batch_size
+
len
(
X
)
print
(
f
"loss:
{
loss
:
>7f
}
[
{
current
:
>5d
}
/
{
size
:
>5d
}
]"
)
def
test_loop
(
dataloader
,
model
,
loss_fn
):
# Set the model to evaluation mode - important for batch normalization and dropout layers
# Unnecessary in this situation but added for best practices
model
.
eval
()
size
=
len
(
dataloader
.
dataset
)
num_batches
=
len
(
dataloader
)
test_loss
,
correct
=
0
,
0
# Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
# also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
with
torch
.
no_grad
():
for
X
,
y
in
dataloader
:
pred
=
model
(
X
)
test_loss
+=
loss_fn
(
pred
,
y
)
.
item
()
correct
+=
(
pred
.
argmax
(
1
)
==
y
)
.
type
(
torch
.
float
)
.
sum
()
.
item
()
test_loss
/=
num_batches
correct
/=
size
print
(
f
"Test Error:
\n
Accuracy:
{
(
100
*
correct
)
:
>0.1f
}
%, Avg loss:
{
test_loss
:
>8f
}
\n
"
)
We initialize the loss function and optimizer, and pass it to
train_loop
and
test_loop
.
Feel free to increase the number of epochs to track the model’s improving performance.
loss_fn
=
nn
.
CrossEntropyLoss
()
optimizer
=
torch
.
optim
.
SGD
(
model
.
parameters
(),
lr
=
learning_rate
)
epochs
=
10
for
t
in
range
(
epochs
):
print
(
f
"Epoch
{
t
+
1
}
\n
-------------------------------"
)
train_loop
(
train_dataloader
,
model
,
loss_fn
,
optimizer
)
test_loop
(
test_dataloader
,
model
,
loss_fn
)
print
(
"Done!"
)
Epoch 1
-------------------------------
loss: 2.311154 [ 64/60000]
loss: 2.289629 [ 6464/60000]
loss: 2.272683 [12864/60000]
loss: 2.262568 [19264/60000]
loss: 2.246720 [25664/60000]
loss: 2.229438 [32064/60000]
loss: 2.230696 [38464/60000]
loss: 2.199316 [44864/60000]
loss: 2.187020 [51264/60000]
loss: 2.158936 [57664/60000]
Test Error:
 Accuracy: 45.9%, Avg loss: 2.150578

Epoch 2
-------------------------------
loss: 2.164762 [ 64/60000]
loss: 2.148163 [ 6464/60000]
loss: 2.096328 [12864/60000]
loss: 2.110341 [19264/60000]
loss: 2.057025 [25664/60000]
loss: 2.003846 [32064/60000]
loss: 2.029172 [38464/60000]
loss: 1.945778 [44864/60000]
loss: 1.947813 [51264/60000]
loss: 1.875144 [57664/60000]
Test Error:
 Accuracy: 52.3%, Avg loss: 1.874600

Epoch 3
-------------------------------
loss: 1.917092 [ 64/60000]
loss: 1.878315 [ 6464/60000]
loss: 1.768980 [12864/60000]
loss: 1.803640 [19264/60000]
loss: 1.695156 [25664/60000]
loss: 1.654540 [32064/60000]
loss: 1.677486 [38464/60000]
loss: 1.575516 [44864/60000]
loss: 1.603369 [51264/60000]
loss: 1.499136 [57664/60000]
Test Error:
 Accuracy: 57.8%, Avg loss: 1.517382

Epoch 4
-------------------------------
loss: 1.595141 [ 64/60000]
loss: 1.551882 [ 6464/60000]
loss: 1.410729 [12864/60000]
loss: 1.474389 [19264/60000]
loss: 1.364348 [25664/60000]
loss: 1.363706 [32064/60000]
loss: 1.382762 [38464/60000]
loss: 1.298951 [44864/60000]
loss: 1.336140 [51264/60000]
loss: 1.243984 [57664/60000]
Test Error:
 Accuracy: 62.9%, Avg loss: 1.263355

Epoch 5
-------------------------------
loss: 1.346288 [ 64/60000]
loss: 1.320302 [ 6464/60000]
loss: 1.162194 [12864/60000]
loss: 1.259876 [19264/60000]
loss: 1.144563 [25664/60000]
loss: 1.169922 [32064/60000]
loss: 1.196316 [38464/60000]
loss: 1.121641 [44864/60000]
loss: 1.164390 [51264/60000]
loss: 1.087258 [57664/60000]
Test Error:
 Accuracy: 64.9%, Avg loss: 1.100687

Epoch 6
-------------------------------
loss: 1.175312 [ 64/60000]
loss: 1.170416 [ 6464/60000]
loss: 0.995274 [12864/60000]
loss: 1.121217 [19264/60000]
loss: 1.004056 [25664/60000]
loss: 1.036032 [32064/60000]
loss: 1.076645 [38464/60000]
loss: 1.005788 [44864/60000]
loss: 1.049191 [51264/60000]
loss: 0.985340 [57664/60000]
Test Error:
 Accuracy: 65.9%, Avg loss: 0.992516

Epoch 7
-------------------------------
loss: 1.053627 [ 64/60000]
loss: 1.070891 [ 6464/60000]
loss: 0.878569 [12864/60000]
loss: 1.026357 [19264/60000]
loss: 0.913745 [25664/60000]
loss: 0.940245 [32064/60000]
loss: 0.996950 [38464/60000]
loss: 0.929135 [44864/60000]
loss: 0.967994 [51264/60000]
loss: 0.916006 [57664/60000]
Test Error:
 Accuracy: 67.3%, Avg loss: 0.917575

Epoch 8
-------------------------------
loss: 0.963031 [ 64/60000]
loss: 1.000980 [ 6464/60000]
loss: 0.794553 [12864/60000]
loss: 0.958322 [19264/60000]
loss: 0.853055 [25664/60000]
loss: 0.869251 [32064/60000]
loss: 0.940442 [38464/60000]
loss: 0.877607 [44864/60000]
loss: 0.908915 [51264/60000]
loss: 0.865586 [57664/60000]
Test Error:
 Accuracy: 68.5%, Avg loss: 0.863106

Epoch 9
-------------------------------
loss: 0.893365 [ 64/60000]
loss: 0.948321 [ 6464/60000]
loss: 0.731461 [12864/60000]
loss: 0.907097 [19264/60000]
loss: 0.809371 [25664/60000]
loss: 0.815217 [32064/60000]
loss: 0.897766 [38464/60000]
loss: 0.841641 [44864/60000]
loss: 0.864545 [51264/60000]
loss: 0.826516 [57664/60000]
Test Error:
 Accuracy: 69.8%, Avg loss: 0.821541

Epoch 10
-------------------------------
loss: 0.837620 [ 64/60000]
loss: 0.905961 [ 6464/60000]
loss: 0.682135 [12864/60000]
loss: 0.867314 [19264/60000]
loss: 0.775932 [25664/60000]
loss: 0.773175 [32064/60000]
loss: 0.863408 [38464/60000]
loss: 0.815205 [44864/60000]
loss: 0.829714 [51264/60000]
loss: 0.794519 [57664/60000]
Test Error:
 Accuracy: 71.1%, Avg loss: 0.788337

Done!
Further Reading
#
Loss Functions
torch.optim
Warmstart Training a Model
Total running time of the script:
(1 minutes 50.391 seconds)
Download
Jupyter
notebook:
optimization_tutorial.ipynb
Download
Python
source
code:
optimization_tutorial.py
Download
zipped:
optimization_tutorial.zip
On this page
PyTorch Libraries
ExecuTorch
Helion
torchao
kineto
torchtitan
TorchRL
torchvision
torchaudio
tensordict
PyTorch on XLA Devices


---


## Save and Load the Model

**Source:** https://docs.pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html

Rate this Page
★
★
★
★
★
beginner/basics/saveloadrun_tutorial
Run in Google Colab
Colab
Download Notebook
Notebook
View on GitHub
GitHub
Note
Go to the end
to download the full example code.
Learn the Basics
||
Quickstart
||
Tensors
||
Datasets & DataLoaders
||
Transforms
||
Build Model
||
Autograd
||
Optimization
||
Save & Load Model
Save and Load the Model
#
Created On: Feb 09, 2021 | Last Updated: Sep 25, 2025 | Last Verified: Nov 05, 2024
In this section we will look at how to persist model state with saving, loading and running model predictions.
import
torch
import
torchvision.models
as
models
Saving and Loading Model Weights
#
PyTorch models store the learned parameters in an internal
state dictionary, called
state_dict
. These can be persisted via the
torch.save
method:
model
=
models
.
vgg16
(
weights
=
'IMAGENET1K_V1'
)
torch
.
save
(
model
.
state_dict
(),
'model_weights.pth'
)
Downloading: "https://download.pytorch.org/models/vgg16-397923af.pth" to /var/lib/ci-user/.cache/torch/hub/checkpoints/vgg16-397923af.pth

 0%| | 0.00/528M [00:00<?, ?B/s]
 7%|▋ | 35.5M/528M [00:00<00:01, 371MB/s]
 14%|█▍ | 74.0M/528M [00:00<00:01, 390MB/s]
 21%|██▏ | 113M/528M [00:00<00:01, 400MB/s]
 29%|██▊ | 152M/528M [00:00<00:01, 285MB/s]
 36%|███▌ | 190M/528M [00:00<00:01, 320MB/s]
 42%|████▏ | 224M/528M [00:00<00:01, 315MB/s]
 49%|████▊ | 256M/528M [00:00<00:00, 314MB/s]
 55%|█████▌ | 292M/528M [00:00<00:00, 333MB/s]
 63%|██████▎ | 331M/528M [00:01<00:00, 354MB/s]
 69%|██████▉ | 366M/528M [00:01<00:00, 356MB/s]
 76%|███████▋ | 404M/528M [00:01<00:00, 367MB/s]
 84%|████████▍ | 443M/528M [00:01<00:00, 379MB/s]
 91%|█████████ | 479M/528M [00:01<00:00, 377MB/s]
 98%|█████████▊| 516M/528M [00:01<00:00, 362MB/s]
100%|██████████| 528M/528M [00:01<00:00, 349MB/s]
To load model weights, you need to create an instance of the same model first, and then load the parameters
using
load_state_dict()
method.
In the code below, we set
weights_only=True
to limit the
functions executed during unpickling to only those necessary for
loading weights. Using
weights_only=True
is considered
a best practice when loading weights.
model
=
models
.
vgg16
()
# we do not specify ``weights``, i.e. create untrained model
model
.
load_state_dict
(
torch
.
load
(
'model_weights.pth'
,
weights_only
=
True
))
model
.
eval
()
VGG(
 (features): Sequential(
 (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
 (1): ReLU(inplace=True)
 (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
 (3): ReLU(inplace=True)
 (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
 (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
 (6): ReLU(inplace=True)
 (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
 (8): ReLU(inplace=True)
 (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
 (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
 (11): ReLU(inplace=True)
 (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
 (13): ReLU(inplace=True)
 (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
 (15): ReLU(inplace=True)
 (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
 (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
 (18): ReLU(inplace=True)
 (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
 (20): ReLU(inplace=True)
 (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
 (22): ReLU(inplace=True)
 (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
 (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
 (25): ReLU(inplace=True)
 (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
 (27): ReLU(inplace=True)
 (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
 (29): ReLU(inplace=True)
 (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
 )
 (avgpool): AdaptiveAvgPool2d(output_size=(7, 7))
 (classifier): Sequential(
 (0): Linear(in_features=25088, out_features=4096, bias=True)
 (1): ReLU(inplace=True)
 (2): Dropout(p=0.5, inplace=False)
 (3): Linear(in_features=4096, out_features=4096, bias=True)
 (4): ReLU(inplace=True)
 (5): Dropout(p=0.5, inplace=False)
 (6): Linear(in_features=4096, out_features=1000, bias=True)
 )
)
Note
be sure to call
model.eval()
method before inferencing to set the dropout and batch normalization layers to evaluation mode. Failing to do this will yield inconsistent inference results.
Saving and Loading Models with Shapes
#
When loading model weights, we needed to instantiate the model class first, because the class
defines the structure of a network. We might want to save the structure of this class together with
the model, in which case we can pass
model
(and not
model.state_dict()
) to the saving function:
torch
.
save
(
model
,
'model.pth'
)
We can then load the model as demonstrated below.
As described in
Saving and loading torch.nn.Modules
,
saving
state_dict
is considered the best practice. However,
below we use
weights_only=False
because this involves loading the
model, which is a legacy use case for
torch.save
.
model
=
torch
.
load
(
'model.pth'
,
weights_only
=
False
)
Note
This approach uses Python
pickle
module when serializing the model, thus it relies on the actual class definition to be available when loading the model.
Related Tutorials
#
Saving and Loading a General Checkpoint in PyTorch
Tips for loading an nn.Module from a checkpoint
Total running time of the script:
(0 minutes 5.771 seconds)
Download
Jupyter
notebook:
saveloadrun_tutorial.ipynb
Download
Python
source
code:
saveloadrun_tutorial.py
Download
zipped:
saveloadrun_tutorial.zip
On this page
PyTorch Libraries
ExecuTorch
Helion
torchao
kineto
torchtitan
TorchRL
torchvision
torchaudio
tensordict
PyTorch on XLA Devices


---
