##  Sound-guided Semantic Video Generation

![Teaser image](https://kr.object.ncloudstorage.com/eccv2022/homepage/title.png)

**Sound-guided Semantic Video Generation**<br>
 
Abstract: *The recent success in StyleGAN demonstrates that pre-trained StyleGAN latent space is useful for realistic video generation. However, the generated motion in the video is usually not semantically meaningful due to the difficulty of determining the direction and magnitude in the StyleGAN latent space. 
In this paper, we propose a framework to generate realistic videos by leveraging multimodal (sound-image-text) embedding space. As sound provides the temporal contexts of the scene, our framework learns to generate a video that is semantically consistent with sound.
First, our sound inversion module maps the audio directly into the StyleGAN latent space. We then incorporate the CLIP-based multimodal embedding space to further provide the audio-visual relationships. Finally, the proposed frame generator learns to find the trajectory in the latent space which is coherent with the corresponding sound and generates a video in a hierarchical manner. 
We provide the new high-resolution landscape video dataset (audio-visual pair) for the sound-guided video generation task. The experiments show that our model outperforms the state-of-the-art methods in terms of video quality. We further show several applications including image and video editing to verify the effectiveness of our method.*


![Method Image](https://kr.object.ncloudstorage.com/eccv2022/homepage/train.png)
## :floppy_disk: Installation
See the official StyleGAN3 repository for installation.

## Dataset
Download the high fidelity landscape dataset from the link.  

## Training.
```
python3 train_video.py
```


## Generation.
```
python3 gen_video.py
```

## Results
Project Page : https://anonymous5584.github.io/
