# skill-of-the-model-using-cross-validation
This Repo demonstate that why we shall never build a model directly , without performing cross-validation on it.

#introduction :- In this notebook, i will demo 2 things
    # 1st: i take a dataset and build a model around it, and will print the accuracy of model over Training set
    
    #Drawback : in this appraoch we never know, that whats the learning capability of the model
    #           as we directly trained it, and got a training & test accuracy of the model.
    #            So we would never know whether it is underlearn or overlearn(Overfitting) until we do not test it on 
    #            other Observation or testData set.
    
    # so thats the probelm with direct Training  of model,
    # and we donot need to jump right away on conclusion that our model is learning good or learning bad.
    
    
    # 2nd : a) In this approach , we would do cross-validation on model, and get a accuracy of the model.
    #       so this value will tell us how much a model can learn when trained on actual Traning dataset.
    #       Therefor CV helps us in knowing the skill of model on the dataset, without inactual training the model on it.

    # once cv done, then we could do actual training of model on dataset, and from the training and test accuracy we could know whether the model
    # is doing underfitting or overfitting.
    
