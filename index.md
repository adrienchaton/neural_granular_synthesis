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

## MODEL ARCHITECTURE

<p align="center"> <img src="figures/architecture.png"> </p>

&nbsp;

## GRANULAR LATENT SPACE

<p align="center"> <img src="figures/granular_space.png"> </p>

2D visualization (with Principale Component Analysis, *PCA*) of a learned grain space over individual pitched notes of orchestral instruments (from *SOL* dataset). Using latent dimensions of a Variational Auto-Encoder (*VAE*) as a substitute for acoustic descriptors allows for an invertible representation. Grains can be synthesized from any position, such as along a continuous free-synthesis path. A waveform is assembled by overlap-add of the grain sequence.

&nbsp;

## SOUND EXAMPLES FROM THE TRAINED MODELS

*This page is under construction, sound examples are being added prior to the reviewing process.*

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

Using a model with sequential embedding and conditioning trained over the *8  Drums* dataset classes, we can sample structured one-shot audio clips for each of the drum classes. Since its embedding is continuous, we can explore a great diversity of sounds. Once the samples are chosen, they can be played-back in realtime with a 8-track step sequencer, without having an actual sample library but only a generative neural network.
