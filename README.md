# Generative Adversarial Nets for Single Image Super-Resolution
This project was done as a part of the course IE 643: Deep Learning - Theory and Practice at IIT Bombay. 

As a part of my initial contributions to this project, I had also implemented a [vanilla GAN] to generate handwritten digits from a noisy latent space. Single image super-resolution (SISR) has been a notoriously challenging ill-posed problem, which aims to obtain a high-resolution output from one of its low-resolution versions. Despite the breakthroughs in accuracy and speed of SISR using faster and deeper convolutional neural networks (CNN), one central problem to recover the finer texture details super-resolving at large upscaling factors remains largely unsolved. With this motivation, I have attempted to implement the idea proposed by [Christian Ledig et al.] to use a GAN comprising of deep convolutional networks for upscaling (4× factor) natural images to produce the then state-of-the-art photo-realistic super-resolved images.

For details about the implementation, results and conclusion, please refer to the [report].
[vanilla GAN]: https://arxiv.org/pdf/1406.2661.pdf
[Christian Ledig et al.]: https://ieeexplore.ieee.org/document/8099502/
[report]: https://github.com/sumanvid97/DL_Project/blob/master/report.pdf
