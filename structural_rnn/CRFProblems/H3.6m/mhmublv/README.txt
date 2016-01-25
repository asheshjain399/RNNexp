Version 1.01 

Code provided by Graham Taylor, Geoff Hinton and Sam Roweis 

For more information, see:
    http://www.cs.toronto.edu/~gwtaylor/publications/nips2006mhmublv
Permission is granted for anyone to copy, use, modify, or distribute this
program and accompanying programs and documents for any purpose, provided
this copyright notice is retained and prominently displayed, along with
a note saying that the original programs are available from our
web page.
The programs and documents are distributed without any warranty, express or
implied.  As the programs were written for research purposes only, they have
not been tested to the degree that would be advisable in any important
application.  All use of these programs is entirely at the user's own risk.

This subdirectory contains files related to learning and generation:

motiondemo.m        Main file for kearning and generation
gaussiancrbm.m      Trains CRBM with Gaussian visible units
binarycrbm.m        Trains CRBM with binary logistic visible units
paramcount.m        Counts the current CRBM parameters
weightreport.m      Visualizes parameters while learning
getfilteringdist.m  After learning the first CRBM,
                    builds minibatches for the next CRBM
gen.m               Generates data from a CRBM with one hidden layer
gen2.m              Generates data from a CRBM with two hidden layers

The Motion subdirectory contains files related to motion capture data: 
preprocessing/postprocessing, playback, etc ...

Acknowledgments

The sample data we have included has been provided by Eugene Hsu:
http://people.csail.mit.edu/ehsu/work/sig05stf/

Several subroutines related to motion playback are adapted from Neil 
Lawrence's Motion Capture Toolbox:
http://www.cs.man.ac.uk/~neill/mocap/

Several subroutines related to conversion to/from exponential map
representation are provided by Hao Zhang:
http://www.cs.berkeley.edu/~nhz/software/rotations/
