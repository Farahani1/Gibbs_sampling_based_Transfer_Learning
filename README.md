# Between-Domain Instance Transition Via the Process of Gibbs Sampling

This project is provided as a quick and easy programing evidence for the
results presented in the this [paper](https://arxiv.org/abs/2006.14538)
 which addresses Transfer Learning problem.

## An overview of the project and the Results.
In this project we show that when the model trained in the source domain
(MNIST dataset) is directly applied to the target domain (MNIST-M dataset)
it could only achieve 0.49 accuracy. Fortunately, when the method proposed
of the  [paper](https://arxiv.org/abs/2006.14538) is employed for Transfer Learning,
specially when the *Temperature* parameter is involved, the accuracy of the prediction
 in the target domain increased to 0.68.

## Requirements
The project is developed based on Python 2.
In this project, [PyDeep](https://pydeep.readthedocs.io/en/latest/) package is utilized
for training and sampling form RBMs. Also TensorFlow version 1.x
 is used for neural network training.


## Contact Info
Please feel free to contact me if you have any comment or question regarding
 the project or the paper.\
Email address: h.sh.farahani@gmail.com

