# RNN-ActivityRecognition

The classification of an RNN is exponentially sensible to a domain of inputs. Which can result problematic if the most relevant input is outside that domain.

By adding recurrent units with different time delays between the input and the output, we make the RNN sensible to a broader input domain. Thus increasing the chances of considering said relevant input. The resulting architecture is denominated D-RNN.

The inspiration of this architectures lies in:
Maximizing the dependency of an output with respect to an input.
Minimizing the size and complexity of the model.

As a first experiment, the network is tested on recognizing the activities on a video, since this task exhibits lengthy sequences and has practical value. D-RNN is compared with an LSTM, a more complex architecture. Results showcase both have similar perfomance. However, D-RNN has 98% less parameters than an LSTM. The following video showcases the conclusion: https://www.youtube.com/watch?v=piEGvbbbbps&t=503s

# Future work:
- Validate if DRNN is indeed expanding the input domain the output is sensible to.
- Compare DRNN to similar architectures (e.g. NARX RNNs).
- Making the delay of the units a trainable parameter.

# Note:
thesis_V1_6_es.pdf elaborates intuitively on neural networks and builds the basis for the development of D-RNN. In the appendix there is a tutorial on TensorFlow along with a Jupyter-notebook to create the presented Neural Networks. The current version is in spanish, but I'm working on the english one.

