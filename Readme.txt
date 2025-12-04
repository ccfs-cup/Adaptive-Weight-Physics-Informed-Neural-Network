1、SA-PINNs-master is the reference code for the adaptive weight optimization method proposed in the paper. To understand the adaptive weight optimization specifically, focus on the sections where the weights parameter is configured and applied.

2、UNPINN corresponds to the code for simulating two-dimensional heterogeneous unconfined aquifers. The K distribution is the same for both S1 and S2. The boundary condition settings mainly differ at the left and right boundaries - both .py files read from the same K configuration file.

3、GWPINN corresponds to the training code for the middle reaches of the Heihe River Basin. You can set a certain proportion of PDE sampling points so that the memory usage is manageable for the server.