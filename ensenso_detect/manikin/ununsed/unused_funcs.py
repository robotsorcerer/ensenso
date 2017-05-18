# May 18, 2017:: 9:59am
# This was the lone regressor extracted from the original classifier in
# the regressor code. We have phased this in to the trainClassifierRegressor
# code in /manikin/main.py
def trainRegressor(args, resnet, bbox_loader):
    r"""
    Following the interpretable learning from self-driving examples:
    https://arxiv.org/pdf/1703.10631.pdf   we can extract the last
    feature cube x_t from the resnet model as a set of L = W x H
    vectors of depth D, and stack a regressor module to obtain
    bounding boxes
    """
    #hyperparameters
    numLayers, seqLength = 2, 5
    noutputs, lr = 12, args.rnnLR
    inputSize, nHidden = 128, [64, 32]
    batchSize, maxIter   = args.cbatchSize, args.cmaxIter

    #extract feture cube of last layer and reshape it
    res_classifier, feature_cube = None, None
    if args.classifier:    #use pre-trained classifier
      res_classifier = ResNet(ResidualBlock, [3, 3, 3])
      res_classifier.load_state_dict(torch.load('models225/' + args.classifier))
      #freeze optimized layers
      for param in res_classifier.parameters():
          param.requires_grad = False
    else:
        res_classifier = resnet

    #extract last convolution layer
    last_layer, feat_cube = res_classifier.layer3, []
    for param in last_layer.parameters():
        if param.dim() > 1:  # extract only conv cubes
            feat_cube.append(param)

    # for i in range(len(feat_cube)):
    #     print(feat_cube[i].size())

    # print('b4 softmax: ', len(feat_cube))
    '''
    get the soft attention mechanism's x_t vector::
    see this:
    https://arxiv.org/pdf/1511.04119.pdf
    '''
    xt = []
    for x in xrange(len(feat_cube)):
        temp = softmax(feat_cube[x])
        xt.append(temp)
        # print(xt[x].size())

    # print('after softmax: ', len(xt))


    time.sleep(100)
    #accummulate all the features of the fc layer into a list
    for p in res_classifier.fc.parameters():
        params_list.append(p)  #will contain weights and biases
    params_weight, params_bias = params_list[0], params_list[1]
    #reshape params_weight
    params_weight = params_weight.view(128)

    X_tr = int(0.8*len(params_weight))
    X_te = int(0.2*len(params_weight))
    X = len(params_weight)

    #reshape inputs
    train_X = torch.unsqueeze(params_weight, 0).expand(seqLength, 1, X)
    test_X = torch.unsqueeze(params_weight[X_tr:], 0).expand(seqLength, 1, X_te+1)
    # Get regressor model and predict bounding boxes
    regressor = StackRegressive(res_cube=res_classifier, inputSize=128, nHidden=[64,32,12], noutputs=12,\
                          batchSize=args.cbatchSize, cuda=args.cuda, numLayers=2)

    #initialize the weights of the network with xavier uniform initialization
    for name, weights in regressor.named_parameters():
        #use normal initialization for now
        init.uniform(weights, 0, 1)

    if(args.cuda):
        train_X = train_X.cuda()
        test_X  = test_X.cuda()
        # regressor = regressor.cuda()

    #define optimizer
    optimizer = optim.SGD(regressor.parameters(), lr)

    # Forward + Backward + Optimize
    targ_X = None

    for _, targ_X in bbox_loader:
        targ_X = targ_X

    if args.cuda:
        targ_X = targ_X.cuda()

    for epoch in xrange(maxIter):
        for i in xrange(targ_X.size(1)*10):
            inputs = train_X
            targets = Variable(targ_X[:,i:i+seqLength,:])

            optimizer.zero_grad()
            outputs = regressor(inputs)
            #reshape targets for inputs
            targets = targets.view(seqLength, -1)
            loss    = regressor.criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            if epoch % 5 == 0 and epoch >0:

                lr *= 1./epoch
                optimizer = optim.SGD(regressor.parameters(), lr)
            print('Epoch: {}, \tIter: {}, \tLoss: {:.4f}'.format(
                epoch, i, loss.data[0]))

            if i+seqLength >= int(targ_X.size(1)):
                break

    # #initialize the weights of the network with xavier uniform initialization
    # for name, weights in regressor.named_parameters():
    #     #use normal initialization for now
    #     init.uniform(weights, 0, 1)

    # #extract last convolution layer
    # last_layer, feat_cube = res_classifier.layer3, []
    # for param in last_layer.parameters():
    #     if param.dim() > 1:  # extract only conv cubes
    #         feat_cube.append(param)
    # '''
    # get the soft attention mechanism's x_t vector::
    # see this:
    # https://arxiv.org/pdf/1511.04119.pdf
    # '''
    # lt = []  # this contains the soft max
    # for x in xrange(len(feat_cube)):
    #     temp = softmax(feat_cube[x])
    #     lt.append(temp)
    #
    # # find xt = Sum_i^(KxK) l_t_i X_t_i
    # xt = []
    # for i in xrange(len(feat_cube)):
    #     temp = torch.mul(lt[i], feat_cube[i])
    #     xt.append(temp)
    #     print(xt[i].size())
    #
    # # Now feed each tensor in xt through LSTM layers
    # '''
    # feat cube is of shape
    # 64L, 32L, 3L, 3L
    # 64L, 64L, 3L, 3L
    # 64L, 32L, 3L, 3L
    # 64L, 64L, 3L, 3L
    # 64L, 64L, 3L, 3L
    # 64L, 64L, 3L, 3L
    # 64L, 64L, 3L, 3L
    # '''
