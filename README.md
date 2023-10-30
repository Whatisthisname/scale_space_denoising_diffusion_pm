# Scale-space DDPM Training and Inference
In this university project, I experimented with different ways of more efficiently training and sampling from DDPMs by decomposing the usual forward process into two halves with differing resolutions, as the initial stages (probably) do not need the full detailed representation to get going. Results were mixed, mostly because training DDPMs is an art form that requires very steady hands to ensure crisp samples. The associated report is in this repository as well.

To run all the experiments described, run `make full` from the root of the repository and go get some lunch.
