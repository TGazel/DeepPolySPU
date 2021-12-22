# RIAI 2021 Project Report 

This report documents and explains on a high-level our solution to
building a verifier for SPU activation with DeepPoly, in the framework of the RTAI course.  

The authors of this solution are 
Ghjulia Sialelli, Kouroche Bouchiat and Thomas Gazel-Anthoine.

## Solution structure

The file `DeepPoly.py` contains the encoding of DeepPoly 
using a neural network architecture with PyTorch.
DP_Shape objects store DeepPoly constraints and bounds and are 
the objects on which the neural network operates. 
DeepPoly transformers for Linear, Flatten, Normalization, ReLU and SPU layers 
are built as `nn.Module`. 

The chosen shape for the DeepPoly transformation of the SPU layer is described and can be interacted with 
in the `DP_SPU.ggb` GeoGebra file. Three of the constraints boundaries are parametrized 
(upper constraint in the case u<0, lower constraint in the case l>0 and the crossing case).
These parameters can be modified in the GeoGebra file and are optimized in our solution.
While in the code the parameters are not restricted in range to avoid gradient problems, in GeoGebra they are in the [0,1] range for readabilty.

To verify the output of some neural network and some input under a given perturbation a DP_verifier is built,
consisting of a DP_net, the abstract twin of the neural network, and a final layer 
that pits output nodes against the true label to check the property is verified.
DP_net transforms the shape described by the input and the perturbation (and by clamping in the MNIST input [0,1] range) 
by passing it through the network and running back substitution before every non-exact (i.e. SPU) transformers 
to achieve the best bounds before applying the transformer.
The network has 3 parameters for each SPU node. The parameters are optimized, using a constant learning rate and a simple loss that focuses only on the worst violation of the property, by the verifier 
until the output is verified, immediately outputting verified, or the maximum number of epochs is reached, in which case it outputs not verified. 
Note that given the definition of the parameters, no more than one parameter per node will be updated at each iteration.

## Further work

We have identified some aspects that may be modified and some possible extensions for our solution:
<ul>
<li> The maximum number of epochs we fix is 1000, which on the test example leaves us well below the one minute limit for our machines.
This limit may be increased, and even set to infinity since we may abuse the project guideline that 
specify that if runtime limit is reached, the output is automatically set to not verified. This would have the advantage 
to use the entirety of the ressources. We decided against that.</li>
<li> The specified loss may not be the best for the task, since it does not consider output node in isolation
and may in rare cases stay stuck in a situation where improving one output worsens another. 
Additionally, the constant learning rate may not be optimal for such situations. Note also that the initialization is random 
in our solution, but a more educated guess may allow for slightly lower running time.</li>
<li> While the three parameters almost cover all possible non-dominated (in an inclusion sense) shapes 
and the network can thus in theory choose between all possible optimal shapes, 
there is a small case (when 0&lt;u&lt; sqrt(0.5)) where the upper constraint 
is neither optimal nor parametrized. There exist in this specific case non-dominated shapes the network
does not have access to. We considered this situation not important, since it does not occur often and should not drastically reduce verifier performance. 
</li>
<li> To avoid NaN in the network optimization, division have additional 1e-6 terms to avoid division by 0. 
While for the test cases the given simple networks should not have this problem (in particular there are no situation where u=l),
one may want to replace such solution with proper check and case handling.</li>
</ul>

