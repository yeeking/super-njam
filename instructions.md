# Jazz solo generator 

I want to create and train a transformer model using PyTorch and lightning that I can use to generate jazz solos in the form of symbolic musical events (note and timbre control messages) using the huggingface transformers API. Once I have a trained model, I would like to have a simple command line program written in C++ that uses a GGUF export of the model  to generate jazz solo 'continuations' of musical data files. This requirement to be able to export the trained model to GGUF/ llama cpp format and run it via a natively compiled C++ program places constraints on the design relating to the hugging face library, tokenisers and so on. 

In the working folder you will find various useful resources:

* a dataset of jazz solos with a readme in the data folder, the weimar jazz dataset. This is the training data. 
* some tutorials showing you how to train llama models in hugging face transformers library from scratch and how to fine tune. 
* a llama.cpp folder which shows you how to work with GGUF models, including python scripts to export to gguf format. 

## Stage 1: Develop a text-based language to represent the jazz solos. Call it NJamV3

To develop the model you will first need to design a compact text representation of the data in the weimar dataset. In the text representation, it is important to represent the expressiveness of the data:

- the pitch of the notes
- timing - time between notes, length of notes 
- loudness/ velocity 
- pitch variation over time
- tonal variation/ timbre information if available

The language needs to be compact in that it should be possible to encode musical phrases in as few tokens as possible, without losing the richness of the data. For example, you might consider combining pitch and loudness information into a single token but to have all combinations of pitch-loudness pairs as unique tokens. It is up to you to figure out how to do this but the objective is to minimise the number of tokens and retain richness - as less tokens means faster performance in training and in inference, both of which are very desirable. 

It should be possible to convert from the weimar jazz database representation as seen in the sqlite3 database wjazzd.db and described other information in this folder into the new NJamV3  jazz solo language. It should be possible to convert from NJamV3 to MIDI data, including note on and off messages, plus expressive pitch and volume control pitch bend and CC messages. It should also be possible to translate back from MIDI data of that sort to NJamV3. 

### Outputs of Stage 1: design the language

* Description of the NJamV3  language
* Code to convert from weimar data to NJamV3
* Code to convert from NJamV3 to MIDI data
* Code to convert from MIDI data to NJamV3

## Stage 2: Develop an efficient tokeniser that uses as few tokens as possible to represent the NJamV3  text jazz solo language

Building on the text representation you develop of the musical data, you should think of a gguf/ transformers compatible way of designing an efficient tokenizer. As part of the development, I would like you to develop some tests of different methods of tokenisation, from standard simple ones, through things like BPE to a custom, very efficient one. Then it should be possible to compare tokenisations of the same text with the different tokenisers to show that it consistently uses a lot less tokens. 

### Outputs of Stage 3: design the tokenisers

* Code to run tokeniser comparison experiment comparing standard and custom tokenisers and how many tokens they generate
* llama cpp compatible tokenizer in Python 


## Stage 3: Training process

Once you have designed the language for the jazz data and you have a plan for the tokenisation, I want you to work on the training script. The training script should make it easy to create models of different sizes, architecture settings (all based on the standard, GGUF exportable huggingface causalLM model classes)  and with different tokenisers. I would like to be able to do a big training run with many models to compare their performance. Use the standard metrics for training signals and best practice around training models of this size and with this kind of data. Remember that the models are learning the text NJamV3 language. Use tensor board to log all the training signals such as training and validation error, perplexity and other standard metrics using when training LLMs. You might want to use lightning to make it easier to manage the training devices, and to add more customisations. 

Important features of the training script: 

* Custom rendering during training - I want to see how the model is performing in terms of the kind of musical sequences it is generating. Therefore, we will need a way to render out audio examples and 'scores' of outputs - so to have say, 4 examples from the dataset and to generate continuations of those examples. This should be done via midi rendering (NJamV3 -> midi -> audio) 

* Ability to do a hyper parameter exploration with different model settings (number of layers, attention heads etc.)

* Logging to tensorboard including test audio outputs   play-able in tensor board 

### Outputs of Stage 3: training script

* Training script 
* Hyper-parameter / model architecture exploration possible
* Custom logging of audio and midi continuations of jazz solo prompts to tensor board 


## Stage 4: Exporting and testing the GGUF

I have created a folder called cplusplus/llamacpp-minimal-example  which contains a C plus plus project you can adapt and use to test the export gguf models. Use cmake to build and test it. Verify against the python code. 

I would then like to be able to do some performance testing comparing different quants of the models, so in this stage, I would like to generate models with a range of different quantitations and compare their 'tokens per second' and 'profill' performance. 

### Outputs of Stage 4: export and test

* cmake project that can build a command line program that loads a gguf model then uses it to create a continuation in NJamV3 format of a text file, also in NJamV3.

## General guidelines

I have created a file called coding_guidelines.md which provides general guidelines on how to write code and manage the project. 



