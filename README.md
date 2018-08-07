# RNN-ActivityRecognition

The classification of an RNN is exponentially sensible to a domain of inputs. Which can result problematic if the most relevant input is outside that domain.

By adding recurrent units with different time delays between the input and the output, we make the RNN sensible to a broader input domain. Thus increasing the chances of considering said relevant input. In the video this architecture is D-RNN.

The inspiration of this architectures lies in:
Maximizing the dependency of an output with respect to an input.
Minimizing the size and complexity of the model.

As a first experiment, the network is tested on recognizing the activities on a video, since this task exhibits lengthy sequences and has practical value.

# Results
The following video shows that DRNN achieves similar performance to an LSTM (Popular architecture to deal with the sensibility of inputs) but with 98% less parameters.  DRNN is then a plausible architecture when performance could be sacrificed for model size.

<a href="http://www.youtube.com/watch?feature=player_embedded&v=piEGvbbbbps
" target="_blank"><img src="http://img.youtube.com/vi/piEGvbbbbps/0.jpg" 
width="240" height="180" border="10" /></a>

# Future work ideas:
- Validate if DRNN is indeed expanding the input domain the output is sensible to.
- Compare DRNN to similar architectures (e.g. NARX RNNs).
- Making the delay of the units a trainable parameter.

# Usage
See documentation.pdf

# Neural Networks and TensorFlow Tutorial
thesis_V1_6_es.pdf elaborates intuitively on neural networks and builds the basis for the development of D-RNN. In the appendix there is a tutorial on TensorFlow along with a Jupyter-notebook to create the presented Neural Networks. The current version is in spanish, but I'm working on the english one.
