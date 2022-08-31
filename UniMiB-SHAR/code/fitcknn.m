classdef fitcknn < handle
properties
  BreakTies; % 'nearest', 'random', 'smallest'
  CategoricalPredictors;
  ClassNames;
  Cost; 
    % Cost(i,j) is the cost of classifying an observation into class j 
    % if its true class is i. 
    % By default, Cost(i,j)=1 if i~=j, and Cost(i,j)=0 if i=j.
  Distance;
  DistanceWeight; % 'equal', 'inverse', 'squaredinverse', @fcn
  DistParameter;
  ExpandedPredictorNames;
  IncludeTies = false;% Logical value indicating whether predict includes all the neighbors whose distance values are equal to the Kth smallest distance. If IncludeTies is true, predict includes all these neighbors. Otherwise, predict uses exactly K neighbors (see 'BreakTies').
  ModelParameters;
  Mu;
  NumNeighbors; % default behaviour: single nearest neighbor
  NumObservations; % number of observations = number of rows
  X;
  Y;
  PredictorNames ;
  Prior; 
    % Numeric column vector of prior probabilities for each class. 
    % The order of the elements of Prior corresponds to the order of the classes
    % in ClassNames.
    % DEFAULT : all classes have equal probabilies
    % MOTIVATION : we might have collected a lot of data of a rare disease and not much 
    %   data of a common disease. We want the model to know that one class is rarer than another
    
  ResponseName; % String describing the response variable Y.
  Sigma;
  W; % Numeric vector of nonnegative weights with the same number of rows as Y. Each entry in W specifies the relative importance of the corresponding observation in Y.
end
methods
  function obj = fitcknn(XX,YY)
  % INPUT ARGUMENTS
  % XX : training data
  %   format : each column in XX represents one feature variable. 
  %            each row in XX represents one training point
  %
  % YY : Class labels
  %   format : single column vector. Each entry (row) corresponds to the label
  %            of the corresponding row in XX
  %   Currently only accepting integer labels from 1 to NumClasses 1
    obj.NumNeighbors = 1;
    obj.NumObservations = size(XX)(1);
    obj.X = XX;
    obj.Y = YY;
    obj.ClassNames = unique(YY);
    obj.Cost = ones(length(obj.ClassNames),length(obj.ClassNames)) - ... 
      eye(length(obj.ClassNames));
    obj.BreakTies = 'smallest';
    obj.Prior = ones(size(obj.ClassNames))/length(obj.ClassNames);
  endfunction

  function [label,score,cost] = predict(mdl,Xnew)
  % INPUT ARGUMENTS
  % mdl: classification model. equivalent to "this"
  %
  % Xnew: observation points which we want to classify
  %   format: N x n matrix, where 
  %           N is the number of observations to classify and
  %           n is the number of training features
  %
  % OUTPUT ARGUMENTS
  % label : predicted class label
  %
  % score : predicted class score or posterior probabilities
  %   format : N x K matrix, where
  %            N is the number of observations to classify and
  %            K is the number of classes
  %
  % cost : expected costs
  %   format : N x K matrix, where
  %            N is the number of observations to classify and
  %            K is the number of classes
  %
  % IMPLEMENTATION
  % For each class, the probability of classifying the observation as such is
  % computed and stored in score. The observation is a classified by the class
  % largest with the largest score. From the score array, we compute the 
  % classification cost array. The observation is a classified by the class
  % with the smallest classification cost.
  % 
  % 
  % A training point is identified by its row number in mdl.X and mdl.Y
  % 
  
    % Verify inputs
    assert(size(Xnew)(2) == size(mdl.X)(2)); % observation has same number of features as training set
    
    % Initialize output
    label = zeros(size(Xnew)(1),1);
    score = zeros(size(Xnew)(1),length(mdl.ClassNames));
    cost = zeros(size(Xnew)(1),length(mdl.ClassNames));
    

    for i = 1:size(Xnew)(1)% Think of a better way to parallelize
    % Similarity Measure / Distance
      % distance of training points to observation point
      % Input : Xnew, mdl.X
      % Output : distances
      Xnew_row = Xnew(i,:);
      assert((size(Xnew_row))(1)==1); % Xnew is a row matrix
      % Similarity Measure
      difference_vectors = mdl.X - Xnew_row ; % subtract observed row vector from every row in training set
      % format : M x n, where
      distances = norm(difference_vectors,"rows"); % norm all the feature differences for each training example (row)
      %disp(distances);


      
      % Nearest Neighbors
        % training points closest to observation point
        % Input : distances, mdl.NumNeighbors
        % Output : nearest_neighbors_id, nearest_neighbors_classes
      distance_with_id = [distances, transpose(1:mdl.NumObservations)]; % add index as a second column
      sorted_distances_with_id = sortrows(distance_with_id,1); % sort the similarity measures
        % Example : [4.123, 1; 5.123, 2; 4.321, 3]
      nearest_neighbors_id = sorted_distances_with_id(1:mdl.NumNeighbors,2); % obtain the k nearest neighbours
        % indices of training points ranked by distances
        % Example : [423, 123, 152, 245]
      nearest_neighbors_classes = mdl.Y(nearest_neighbors_id); % lookup classes of k nearest neighbours
        % array is ranked by distance
        % Example : [3; 2; 3; 2; 1;]

      % Classification - weights (distance)
        % weight of every training point by distance
        % Input : mdl.DistanceWeight, distances
        % Output : weights
      if (strcmp(mdl.DistanceWeight,'inverse'))
        weights = distances.^(-1);
      elseif (strcmp(mdl.DistanceWeight,'squaredinverse'))
        weights = distances.^(-2);
      else % (mdl.DistanceWeight == 'equal')
        weights = ones(size(distances));
      end
      
      % Classification - weights (prior)
        % weight of every training point by prior knowledge
        % Input : mdl.DistanceWeight, distances
        % Output : weights
      weights = weights .* mdl.Prior(mdl.Y);
      
      % Edge Case
        % What if observation exactly equal to one of the training points?
        % -> ignore the point, otherwise inverse is crazy
      weights(!isfinite(weights)) = 0;

      % Posterior Probability
        % posterior probability = (relevant weights) / (total weights)
        % Input : numClasses, weights, nearest_neighbors_id
        % Output : score
      for j = 1:length(mdl.ClassNames);
        
        nearest_neighbors_weights = weights(nearest_neighbors_id); 
          % weight of training points in nearest neighbor ball
        relevant_neighbor_id = nearest_neighbors_classes == j; 
          % if we classify observation as j, which weights should we consider
        relevant_score = sum(nearest_neighbors_weights(relevant_neighbor_id)); 
        tot_score = sum(nearest_neighbors_weights); 
        score(i,j) = relevant_score/tot_score;

      end
%        score(i,:)
%        nearest_neighbors_classes
%        nearest_neighbors_weights
%        tot_score
%        score

      assert(!any(isnan(nearest_neighbors_weights)));
      assert(any(score(i,:)));
  
      % Costs ; think of a way to vectorize
        % classification cost
        % Input : score(i,:), mdk.Cost
        % Output : cost(i,:)
      for j = 1:length(mdl.ClassNames); % number of classes
        % expected cost
        y_hat = sum(score(i,:) .* mdl.Cost(j,:));
        cost(i,j) = y_hat;
      end
      
      % Detect ties
        % ties_exists when there are multiple minimum
        % Input : cost(i,:),
        % Output : ties_exists
      min_cost = min(cost(i,:)); % unique costs values
      ties_exists = false;
      if (nnz(cost(i,:) == min_cost)>1)
        ties_exists = true;
      end
      
      % Classification 
        % Input : ties_exists, mdl.BreakTies, cost(i,:)
        % Output : label(i)
      if (ties_exists)
        % find which classes are tied
        tied_classes = find(cost == min(cost(i,:)));
          % Example : [2,6,9]
        if (strcmp(mdl.BreakTies, 'random'))
          % randomly select a class from those that are tied
          label(i) = tied_classes(randi(length(tied_classes)));
          
        elseif (strcmp(mdl.BreakTies, 'nearest'))
          % find first occurance of each class in distance rank
          first_instances = zeros(size(tied_classes));
          for j = 1:length(tied_classes)
            first_instances(j) = find(nearest_neighbors_classes == tied_classes(j))(1);
          end
          first_instances = [first_instances; tied_classes];
          first_instances = sortrows(first_instances,1);
          
          label(i) = first_instances(2);
        else % (mdl.BreakTies == 'smallest')
          [sorted_costs, sorted_classes] = unique(cost(i,:));
          label(i) = sorted_classes(1);
        endif
      else
        [~,label(i)] = min(cost(i,:));
      endif
    end
  endfunction
  
  function L = loss(mdl,X,Y)
  % DESCRIPTION
  % returns a scalar representing how well mdl classifies the data in X, 
  % when Y contains the true classifications.
  % When computing the loss, loss normalizes the class probabilities in Y 
  % to the class probabilities used for training, stored in the 
  % Prior property of mdl.
  
  % INPUT
  % X : observations
  %   format : number of columns = number of features
  %            number of rows = number of new observations
  % Y : true classes of observations X
  %   format : same number of rows as X. 
  %            Each row of Y represents the classification of the corresponding row of X.
  %
  % OUTPUT
  % L : classification loss. precise meaning depends on weights and loss function
  %   Default: 'lossfun' = 'classiferror'
  %
  % EXPLANATION
  % 'classiferror' is the weighted fraction of misclassifications of observations X
  % The fraction of misclassification is the number of misclassifed observations 
  % in X over the number of observations in X.
  % The *weighted* fraction counts misclassified observations in X normalized 
  % by "how important that observation". Less common classes are given less weight.
  % Weights are found by two normalizations. Normalizing observations according 
  % to class priors reduces the importance of "rare diseases"
  % The second normlization brings the result between 0 and 1 so that we can interpret the results.
  
  % weights : observation weights
  %   how important each observation is
  %   by default all observations are equally important. weights = ones(size(X,1),1)
  
    assert (size(X)(1) == size(Y)(1));
    [predicted_labels, predicted_scores] = mdl.predict(X);
    % User-defined weight ratio
    weights_ratio = ones(size(Y,1),1); % TODO: handle other weight_ratios
    assert(size(mdl.Prior) == size(Y));
    weights_ratio_prior = weights_ratio .* mdl.Prior(Y);
    weights = weights_ratio_prior / sum(weights_ratio_prior);
    misclassifications = (predicted_labels != Y);
    misclassified_observation_weights = weights(misclassifications);
    L = sum(misclassified_observation_weights);
    assert(L);
    assert(length(L) == 1);
  endfunction
%  function display 
end
end
