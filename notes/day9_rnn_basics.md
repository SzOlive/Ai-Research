## Day 9: RNN (Recurrent Neural Network) - Complete Theory Guide
### Video links:
https://youtu.be/AsNTP8Kwu80?si=auMXrCnMD8Z7R2lw

https://youtu.be/UNmqTiOnRfg?si=qJiBtjtbg8Jt_Oze

### RNN summary
```
Important ideas:

RNN: Neural network for sequential data.

Hidden State: Memory that flows across time.

Time Steps: Each element of sequence. In text, each word is a time step.

RNN Applications
Task	              Example
Text generation	      GPT ancestors
Speech recognition	  voice assistants
Machine translation	  Google Translate (old)
Time series	          stock prediction
RNN (Recurrent Neural Network)

Designed for sequential data.Sequencial data is data where the order matters like text, audio, video, etc.The model needs to remember the past inputs.

Key idea:
Model keeps memory using hidden state.

Equation:
h_t = f(Wx_t + Uh_{t-1})

Where:
x_t = input
h_t = hidden state

Diagram:

x1 → RNN → h1
x2 → RNN → h2
x3 → RNN → h3

But internally:

h1 = f(x1)
h2 = f(x2 + h1)
h3 = f(x3 + h2)

So past information flows forward.

Applications:
- NLP (natural language processing)
- speech
- time series 
```
### 1. What is an RNN?
RNN stands for Recurrent Neural Network. It's a type of neural network designed specifically for sequential data where order and context matter.

Simple Definition:
Unlike regular neural networks (MLP, CNN) that process one input independently, RNNs have memory - they remember what they've seen before and use that context to understand current input.

The Memory Analogy:
```
Reading a book:
- You don't understand page 50 without having read pages 1-49
- Each new page builds on what you've learned before
- Your understanding evolves as you read more
```
Same as for stock market prediction where you need to remember the past to predict the future.

RNN does the same with sequences!
### 2. Why Do We Need RNNs? (Limitations of Other Networks)
MLP/CNN Problem with Sequences:
```
Network	          How it Processes	          Problem with Sequences
MLP	              Each input independent	  "I love" vs "I hate" - same words, opposite meaning! MLP can't capture context
CNN	              Looks at local patterns	  Can see nearby words but loses long-range dependencies

Example: Understanding Context
Sentence 1: "The movie was not good, I fell asleep"
Sentence 2: "The movie was good, I loved it"

Word: "good"
- In sentence 1: Actually means BAD (because of "not" before it)
- In sentence 2: Means GOOD

MLP/CNN: Would see "good" and think it's positive in BOTH cases!
RNN: Remembers the "not" from earlier, understands the true meaning!
```
### 3. The Core Idea: Hidden State (Memory)
Visualizing the RNN's Memory:
```  
Time t=1:    Time t=2:     Time t=3:     Time t=4:
Input: "I"   Input: "love" Input: "deep" Input: "learning"
   ↓            ↓            ↓             ↓
┌─────┐      ┌─────┐       ┌─────┐       ┌─────┐
│ RNN │──►   │ RNN │──►    │ RNN │──►    │ RNN │
└─────┘      └─────┘       └─────┘       └─────┘
   ↓            ↓            ↓             ↓
Memory:       Memory:       Memory:        Memory:
["I"]      ["I","love"]  ["I","love",    ["I","love",
                           "deep"]         "deep","learning"]
The Hidden State Equation:

h_t = tanh(W1·x_t + U·h_{t-1} + b)

Where:
- h_t = Current memory (hidden state for next time step)
- h_{t-1} = Previous memory (output from previous time step or input)
- x_t = Current input
- W = Weight for current input
- U = Weight for previous memory
- b = Bias for current input
- tanh = Activation function of current memory or input (squashes values between -1 and 1)
```
### 4. How RNN Processes Sequences Step by Step
Step-by-Step Walkthrough:
```
Sequence: "I love AI"

Initial State: h₀ = [0, 0, ..., 0] (empty memory)

Step 1: Process "I"
h₁ = tanh(W·"I" + U·h₀)
Memory now: Contains info about "I"

Step 2: Process "love"
h₂ = tanh(W·"love" + U·h₁)
Memory now: Contains info about "I" + "love"

Step 3: Process "AI"
h₃ = tanh(W·"AI" + U·h₂) [Final step or hidden state]
Memory now: Contains full context of entire sentence!

Final Output: Based on h₃ (understanding of complete sentence)
What the Network Learns:
W: How to interpret new words

U: How to update memory (previous hidden state) with new information(current input or hidden state)

Both: Trained to capture meaningful patterns in sequences
```
### 5. RNN Architecture Diagram
```
                    ╔════════════════════════════════╗
                    ║     UNFOLDED RNN OVER TIME     ║
                    ╚════════════════════════════════╝

                    t=1           t=2           t=3
                    ┌───┐         ┌───┐         ┌───┐
             ┌─────►│ h₁│◄────────│ h₂│◄────────│ h₃│
             │      └─┬─┘         └─┬─┘         └─┬─┘
             │        │             │             │
           ╔═╧══╗   ╔═╧══╗        ╔═╧══╗        ╔═╧══╗
           ║ RNN║   ║ RNN║        ║ RNN║        ║ RNN║
           ╚═╤══╝   ╚═╤══╝        ╚═╤══╝        ╚═╤══╝
             │        │             │             │
             │     ┌──┴──┐       ┌──┴──┐       ┌──┴──┐
             │     │ x₁  │       │ x₂  │       │ x₃  │
             │     │"I"  │       │"love"│      │"AI" │
             │     └─────┘       └─────┘       └─────┘
             │        │             │             │
             └────────┴─────────────┴─────────────┘
                         Memory Flow
Key Components Labeled:

┌─────────────┐
│     xₜ       │ → Input at current time step
├─────────────┤
│     hₜ       │ → Hidden state (memory after current step or calculation)
├─────────────┤
│hₜ = f(xₜ, hₜ₋₁) → RNN cell function (calculation,tanh)
├─────────────┤
│    W, U, b  │ → Learnable parameters
└─────────────┘
```
### 6. Types of Sequential Problems with RNNs 
```
┌─────────────────────────────────────────────────────────┐
│              SEQUENCE PROBLEM TYPES                     │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  One-to-One: Standard MLP (not sequential)              │
│  [Image] → [Label]                                      │
│                                                         │
│  One-to-Many: Image Captioning                          │
│  [Image] → ["A", "dog", "running"]                      │
│                                                         │
│  Many-to-One: Sentiment Analysis                        │
│  ["I", "love", "this"] → [Positive]                     │
│                                                         │
│  Many-to-Many: Machine Translation                      │
│  ["I", "love", "AI"] → ["J", "aime", "l'IA"]            │
│                                                         │
│  Many-to-Many (shifted): Video Classification           │
│  [Frame1, Frame2, Frame3] → [Frame2, Frame3, Frame4]    │
│                                                         │
└─────────────────────────────────────────────────────────┘
```
### 7. Advantages of RNNs
```
Advantage	        Explanation          	              Example
Memory	            Remembers past information	          Understands "not" before "good"
Variable Length	    Can handle any sequence length        Sentences of different sizes
Parameter Sharing	Same weights used at each step	      Efficient, fewer parameters
Context Awareness	Understands word in context	          "Bank" (river) vs "Bank" (money)
Sequential Logic	Captures order dependencies	          "The cat sat on the mat"
```
### 8. Limitations of Simple RNNs
Problem           	  What Happens                       	Why
Vanishing Gradient	  Forgets long-term dependencies	    Gradients become too small over many steps(when weights are small)
Exploding Gradient	  Training becomes unstable	            Gradients become too large(when weights are large)
Short Memory	      Can't remember very long sequences	Only effective for ~10-20 steps
Slow Training	      Can't parallelize(e.g. GPU)	        Must process sequentially(one step at a time)

Visualizing the Vanishing Gradient Problem:
```
Sentence: "I was born in France, ... (50 words later) ... I speak fluent ____"

Simple RNN: Forgets "France" by the end → Predicts "English" ❌
LSTM/GRU: Remembers "France" → Predicts "French" ✅
This is why we need LSTM(long short term memory) and GRU(gated recurrent unit) (next days)!
```
### 9. Real-World Applications of RNNs
```
┌─────────────────────────────────────────────────────────┐
│                RNN APPLICATIONS                         │
├─────────────────────────────────────────────────────────┤
│                                                          
│  📝 Text Generation:                                   
│  Predict next word: "The cat sat on the ___" → "mat"  
│                                                        
│  🎤 Speech Recognition:                                
│  Audio waveform → "Hello, how are you?"                
│                                                        
│  🌍 Machine Translation:                               
│  "I love AI" → "J'aime l'IA"                           
│                                                        
│  📈 Stock Market Prediction:                           
│  Past prices → Future price trend                      
│                                                        
│  🎵 Music Generation:                                   
│  Previous notes → Next note                             
│                                                        
│  📝 Sentiment Analysis:                                 
│  "This movie is amazing!" → Positive                    
│                                                        
│  🏥 Healthcare:                                         
│  Patient history → Disease prediction                   
│                                                        
└───────────────────────────────────────────────────────
```
### 10. RNN vs Other Architectures
```
Aspect	            MLP	         CNN	          RNN
Data Type	        Independent	 Grid (images)	  Sequence
Memory	            None	     Local context	  Long-term
Input Size	        Fixed	     Fixed	          Variable
Parameter Sharing	No	         Yes (filters)	  Yes (time)
Best For	        Tabular data Images	          Text, speech, time series
```
### 11. Mathematical Foundation (Simplified)
Forward Pass Equations:
```
For each time step t = 1 to T:

1. Update hidden state:
   h_t = tanh(W_h · x_t + U_h · h_{t-1} + b_h)

2. Compute output (if needed):
   y_t = W_y · h_t + b_y
Dimensions:
x_t:    (input_size)
h_t:    (hidden_size)
W_h:    (hidden_size × input_size)
U_h:    (hidden_size × hidden_size) [previous hidden state]
b_h:    (hidden_size)[bias]

Batch processing adds batch dimension:
x: (batch, seq_len, input_size)
h: (batch, seq_len, hidden_size)
```
### 12. Memory Aid: RNN in 5 Steps
```
1. START: Initialize memory (h₀ = zeros)
2. LOOP: For each item in sequence
3. UPDATE: h_new = f(current_input, old_memory)
4. OUTPUT: Use h for prediction if needed
5. REPEAT: Until sequence ends

Mnemonic: "Recurrent Networks Never Forget"
- Recurrent: Loops through time
- Networks: Neural network structure
- Never: Maintains memory
- Forget: But simple RNNs do forget (hence LSTM)
```
### 13. Key Terminology Summary
```
Term	            Definition      	           Analogy
Sequence	        Ordered list of data points	   Words in a sentence
Time Step	        One element in sequence	       One word position
Hidden State	    Memory vector	               What you remember so far
Recurrent	        Loop connecting time steps	   Passing notes forward
Unfolding	        Drawing RNN through time	   Showing each step separately
Vanishing Gradient	Forgetting long-term info	   Losing memory of early words
```
### 14. What You'll Learn Next
```
Day 10: RNN (Simple) → Understands sequences but short memory
               ↓
Day 11: LSTM (Long Short-Term Memory) → Remembers for longer
               ↓
Day 12: GRU (Gated Recurrent Unit) → Efficient LSTM alternative
               ↓
Day 13-14: Text Generation → Build actual language model
               ↓
Finally: Transformers → Modern architecture (GPT, BERT)
```
### 15. Quick Reference: PyTorch RNN Parameters
```
python
nn.RNN(
    input_size=10,     # Size of each input vector
    hidden_size=20,    # Size of memory (hidden state)
    num_layers=1,      # Number of stacked RNNs
    batch_first=True,  # Input shape: (batch, seq, features)
    dropout=0,         # Dropout between layers
    bidirectional=False # Process forward and backward
)
Input/Output Shapes:
Input:  (batch, seq_len, input_size)   if batch_first=True
Output: (batch, seq_len, hidden_size)  # All hidden states
Hidden: (num_layers, batch, hidden_size) # Final hidden state
```
### 16. Research Notes Summary
```
┌─────────────────────────────────────────────────────────┐
│              RNN - RESEARCH NOTES                       │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Definition:                                            │
│  Neural network with internal memory for sequences      │
│                                                         │
│  Key Innovation:                                        │
│  h_t = f(W·x_t + U·h_{t-1})  (memory update)            │
│                                                         │
│  Strengths:                                             │
│  • Handles variable-length sequences                    │
│  • Maintains context through time                       │
│  • Parameter sharing across time                        │
│                                                         │
│  Limitations:                                           │
│  • Vanishing gradients (short memory)                   │
│  • Sequential processing (slow)                         │
│  • Difficulty with very long sequences                  │
│                                                         │
│  Applications:                                          │
│  • NLP, speech, time series, any sequential data        │
│                                                         │
│  Next Evolution:                                        │
│  LSTM → GRU → Transformers (solves memory problem)      │
│                                                         │
└─────────────────────────────────────────────────────────┘
```
### 17. Check Your Understanding
✅ Can you answer these?
```
Why can't MLP handle sequences properly?
MLP (Multi-Layer Perceptron) cannot handle sequences properly because it treats each input as independent and ignores order. When you feed a sentence into an MLP, it sees words as separate features without any connection between them. For example, in the sentences "I love AI" and "AI love I", an MLP would see the same bag of words and couldn't tell which order makes sense. MLP also requires fixed-size inputs, but sequences like sentences have variable lengths. Most importantly, MLP has no memory mechanism - it processes each word fresh without remembering what came before, so it misses crucial context like the word "not" changing the meaning of "good" later in the sentence.

What information does hidden state store?
The hidden state stores the network's memory of everything it has seen so far in the sequence. Think of it as a summary vector that captures the important context from all previous time steps. For example, after reading the sentence "I was born in France and have lived there for 20 years", the hidden state would contain information about "France" even though many words have passed. This hidden state gets updated at each step - old information is modified and new information is added, creating a compressed representation of the sequence's meaning up to that point. The size of this memory (hidden_size) determines how much information can be stored.

What does the equation h_t = tanh(Wx_t + Uh_{t-1}) mean?
This equation is how the RNN updates its memory at each time step. Breaking it down: x_t is the current input (like the word you're reading now), and h_{t-1} is the previous memory (what you remembered from earlier). The RNN combines these using two weight matrices: W decides how to interpret the new input, and U decides how to transform the old memory(weight for previous state to this state). These are added together plus a bias, then passed through tanh activation function which squashes values between -1 and 1 to keep the memory stable. The result h_t becomes the new memory, ready for the next time step. This simple formula allows the network to continuously update its understanding as new information arrives.

Why do simple RNNs struggle with long sentences?
Simple RNNs struggle with long sentences because of the vanishing gradient problem during training. When the network tries to learn connections between words far apart (like the first word "France" and the last word "French" in a long sentence), the gradient signal has to flow backwards through many time steps. With each step, this signal gets multiplied by small numbers (less than 1), causing it to shrink exponentially until it becomes effectively zero. This means the network cannot learn to connect distant words - it forgets the beginning of long sentences. Mathematically, if you multiply numbers like 0.9 repeatedly, after 50 steps it's practically zero. This limits simple RNNs to remembering only about 10-20 steps back, making them unsuitable for long documents or conversations.

Give 3 real-world applications of RNNs
1. Text Generation/Autocomplete: When you type on your phone and it suggests the next word, RNNs are predicting what comes next based on the sequence of words you've already typed. Gmail's Smart Compose uses similar technology to complete your sentences.

2. Sentiment Analysis: Companies use RNNs to analyze customer reviews and social media posts to determine if the sentiment is positive, negative, or neutral. The RNN understands context, like recognizing that "not good" is negative even though it contains the word "good".

3. Speech Recognition: Virtual assistants like Siri, Alexa, and Google Assistant use RNNs to convert your spoken words (audio sequences) into text. The RNN processes the audio frame by frame, maintaining context to understand words correctly even with different accents or speaking speeds.
```