# CV-Playground<!-- omit in toc -->
- [Adversarial Attack](#adversarial-attack)

## Adversarial Attack

FGSM (Fast Gradient Sign Attack) attack is a white-box attack with the goal of misclassification.

In this project, we use the FGSM attack to generate adversarial examples for the MNIST dataset.

Here, epsilon is the pixel-wise perturbation amount. As epsilon increases, the adversarial example is more likely to be misclassified.

<img src="AdversarialAttack/accuracy.jpeg">

However, as epsilon increases, the perturbations become more easily to be perceived. There is a tradeoff between accuracy degredation and perceptibility. Here are some examples of sucessful attacks at each epsilon value:

<img src="AdversarialAttack/example.jpeg">

