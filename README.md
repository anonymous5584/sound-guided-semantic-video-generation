##  Sound-guided Semantic Video Generation

![Teaser image](https://kr.object.ncloudstorage.com/eccv2022/homepage/title.jpg)

**Sound-guided Semantic Video Generation**<br>
 
Abstract: *The recent success in StyleGAN demonstrates that pre-trained StyleGAN latent space is useful for realistic video generation. However, the generated motion in the video is usually not semantically meaningful due to the difficulty of determining the direction and magnitude in the StyleGAN latent space. 
In this paper, we propose a framework to generate realistic videos by leveraging multimodal (sound-image-text) embedding space. As sound provides the temporal contexts of the scene, our framework learns to generate a video that is semantically consistent with sound.
First, our sound inversion module maps the audio directly into the StyleGAN latent space. We then incorporate the CLIP-based multimodal embedding space to further provide the audio-visual relationships. Finally, the proposed frame generator learns to find the trajectory in the latent space which is coherent with the corresponding sound and generates a video in a hierarchical manner. 
We provide the new high-resolution landscape video dataset (audio-visual pair) for the sound-guided video generation task. The experiments show that our model outperforms the state-of-the-art methods in terms of video quality. We further show several applications including image and video editing to verify the effectiveness of our method.*


![Method Image](https://kr.object.ncloudstorage.com/eccv2022/homepage/train.png)
## :floppy_disk: Installation
See the official StyleGAN3 repository for installation.

## Dataset
Download the high fidelity landscape dataset from the [link](https://docs.google.com/spreadsheets/d/1FSkigzc7F0SfatTc8YGyVg23uSg6c3kou6e-a6tpB_I/edit?usp=sharing).  

## Training.
Pre-trained StyleGAN3 Generator : link
```
python3 train_sound2video.py
```


## Generation.
Pre-trained Video Generator

Audio encoder : [link](https://kr.object.ncloudstorage.com/eccv2022/weights/audio_inversion_.pth)

Generator : [link1](https://kr.object.ncloudstorage.com/eccv2022/weights/coarseformer_.pth) [link2](https://kr.object.ncloudstorage.com/eccv2022/weights/fineformer_.pth) [link3](https://kr.object.ncloudstorage.com/eccv2022/weights/midformer_.pth)

```
python3 gen_sound2video.py
```

## Results
Project Page : https://anonymous5584.github.io/