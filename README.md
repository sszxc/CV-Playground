# CV-Playground<!-- omit in toc -->
- [Adversarial Attack](#adversarial-attack)

## Adversarial Attack

FGSM (Fast Gradient Sign Attack) attack is a white-box attack with the goal of misclassification.

In this project, I first use the FGSM attack to generate adversarial examples for the MNIST dataset.

Here, epsilon is the pixel-wise perturbation amount. As epsilon increases, the adversarial example is more likely to be misclassified.

<img src="AdversarialAttack/MNIST_accuracy.jpeg" width='50%'>

However, as epsilon increases, the perturbations become more easily to be perceived. There is a tradeoff between accuracy degredation and perceptibility. Here are some examples of sucessful attacks at each epsilon value:

<img src="AdversarialAttack/MNIST_example.jpeg" width='100%'>

After that, I tried the same strategy on the ImageNet dataset. The results are shown below:

<img src="AdversarialAttack/imagenet_accuracy.jpeg" width='50%'>

<img src="AdversarialAttack/imagenet_example.jpeg" width='100%'>