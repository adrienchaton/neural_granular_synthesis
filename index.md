<!--
<a href="audio/Crash_s_note.wav">sample_embed_crash</a>
https://jekyllcodex.org/without-plugin/open-embed/#
-->

<script type="text/javascript"> 
      // Show button
      function look(type){ 
      param=document.getElementById(type); 
      if(param.style.display == "none") param.style.display = "block"; 
      else param.style.display = "none" 
      } 
</script>

&nbsp;

*This page is under construction, sound and video examples are being added prior to the reviewing process. Stay tuned during that period and afetr, more experiments may be uploaded !*

&nbsp;

## MODEL ARCHITECTURE

<p align="center"> <img src="figures/architecture.png"> </p>

&nbsp;

## GRANULAR LATENT SPACE

<p align="center"> <img src="figures/granular_space.png"> </p>

2D visualization (with Principale Component Analysis, *PCA*) of a learned grain space over individual pitched notes of orchestral instruments (from *SOL* dataset). Using latent dimensions of a Variational Auto-Encoder (*VAE*) as a substitute for acoustic descriptors allows for an invertible representation. Grains can be synthesized from any position, such as along a continuous free-synthesis path, and overlap-add into a waveform. A pitch target can be added.

&nbsp;

## SOUND EXAMPLES FROM THE TRAINED MODELS

*All sound examples are raw outputs of the models (besides inputs for the reconstructions and resynthesis), without any kind of audio effect/processing added.*

### RECONSTRUCTIONS

Data reconstructions.

### SAMPLING

Structured one-shot audio clips generated from the recurrent embedding over the *8  Drums* grain space.

* sample conditioned on "Crash": <audio controls><source src="audio/Crash_s_note.wav"></audio>

### FREE-SYNTHESIS

Traversals in the latent grain space of the *10 animals* dataset.

* linear interpolation conditioned on "Cat": <audio controls><source src="audio/cat_interp_seq.wav"></audio>

### DATA-DRIVEN RESYNTHESIS

Audio style transfer.

&nbsp;

## EXAMPLE INTERFACES

With GPU support, for instance a sufficient dedicated laptop chip or an external thunderbolt hardware, the models can be ran in real-time. In order to apply trained models to these different generative tasks, we currently work on some prototype interfaces based on a [Python OSC](https://pypi.org/project/python-osc/) server controlled from a [MaxMsp](https://cycling74.com) patch.

### NEURAL DRUM MACHINE

Using a model with sequential embedding and conditioning trained over the *8  Drums* dataset classes, we can sample structured one-shot audio clips for each of the drum classes. Since its embedding is continuous, we can explore a great diversity of sounds. Moreover, we can alternatively sample random paths to explore other spectro-temporal dynamics. Once the samples are chosen, they can be played-back in realtime with a 8-track step sequencer, without having an actual sample library but only a generative neural network to write the buffers.

[Screenshot](https://raw.githubusercontent.com/anonymized124/neural_granular_synthesis/master/figures/screen_drum_machine.png)

The upper integer box selects the drum condition sent to the model. The trigger to the message **sample_note** generates a drum hit from the recurrent embedding. The trigger to the message **sample_path** draws a randomized linear path in the grain space, of same length that the embedding sample. This can be seen in the python console on the left side, answering to the messages sent through OSC. While the step sequencer loops, it is possible to resample some drum sounds that overwrite the previous buffer.

<video id="drum_machine" class="video-js vjs-default-skin" controls preload="auto" width="600" height="288" data-setup="{}">
<source src="video/drum_machine.mp4" type='video/mp4'>
</video>
