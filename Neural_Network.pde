class NeuralNetwork{
  
  //Our Neural Networks Brain
  int inputL;
  int hiddenL;
  int outputL;
  
  //Learning rate this will effect how large the changes are when training
  float lr = .25;
  
  
  Matrix biash;
  Matrix biaso;
  Matrix weightsih; 
  Matrix weightsho;
  
  //initialize when created
  NeuralNetwork(int i, int h, int o){
    inputL = i;
    hiddenL = h;
    outputL = o;    
    
    weightsih = new Matrix(hiddenL,inputL);
    weightsih.randomize();
    
    weightsho = new Matrix(outputL, hiddenL);
    weightsho.randomize();
    
    biash = new Matrix(hiddenL, 1);
    biash.randomize();
    biaso = new Matrix(outputL, 1);
    biaso.randomize();
  }
  
  
  //Calculate output with the given inputs
  float[] feedForward(float[] input){
       
    Matrix inputs = new Matrix(input.length, 1);
     
    
    for(int i = 0; i < input.length; i++){
      inputs.data[i][0] = input[i];
    }
    
    Matrix hidden = weightsih.mProduct(inputs);
    
    hidden.mAdd(biash);
    
    for(int i = 0; i < hidden.rows; i++){
      for(int j = 0; j < hidden.cols; j++){
        hidden.data[i][j] = sigmoid(hidden.data[i][j]);
      }
    }
    

    
    Matrix output = weightsho.mProduct(hidden);
    output.mAdd(biaso);
    
     
    
    for(int i = 0; i < output.rows; i++){
      for(int j = 0; j < output.cols; j++){
        output.data[i][j] = sigmoid(output.data[i][j]);
      }
    }
       
  
    
    
    float[] done = new float[output.rows*output.cols];
    
    int count = 0;
    
    for(int i = 0; i < output.rows; i++){
      for(int j = 0; j < output.cols; j++){
        done[count] = output.data[i][j];
        count++;
      }
    }
    
    return done;
  }
  
  
  //train the network with expected values for given inputs
  void train(float[] input,float[] answer){
     Matrix inputs = new Matrix(input.length, 1);
    for(int i = 0; i < input.length; i++){
      inputs.data[i][0] = input[i];
    }
    
    Matrix hidden = weightsih.mProduct(inputs);
    hidden.mAdd(biash);
    

    for(int i = 0; i < hidden.rows; i++){
      for(int j = 0; j < hidden.cols; j++){
        hidden.data[i][j] = sigmoid(hidden.data[i][j]);
      }
    }
    
    
    Matrix output = weightsho.mProduct(hidden);
    output.mAdd(biaso);
    
    for(int i = 0; i < output.rows; i++){
      for(int j = 0; j < output.cols; j++){
        output.data[i][j] = sigmoid(output.data[i][j]);
      }
    }
            
    Matrix answers = new Matrix(answer.length,1);
    for(int i = 0; i < answer.length; i++ ){
      answers.data[i][0] = answer[i]; 
    }
    
 
    Matrix oErrors = output.subtract(answers);
     
    Matrix hoT = weightsho.transpose();
    Matrix hErrors = hoT.mProduct(oErrors);
       
    
    Matrix gradient = output;
    for(int i = 0; i < gradient.rows; i++){
      for(int j = 0; j < gradient.cols; j++){
        gradient.data[i][j] = dsigmoid(output.data[i][j]);
      }
    }
    
    gradient.mMult(oErrors);
    gradient.mMult(lr);
    
         
    Matrix hiddenT = hidden.transpose();
    Matrix weighthoD = gradient.mProduct(hiddenT);
    
    
    weightsho.mAdd(weighthoD);
    biaso.mAdd(gradient);
    
    Matrix hgradient = hidden;
    
     for(int i = 0; i < hidden.rows; i++){
      for(int j = 0; j < hidden.cols; j++){
        
        hgradient.data[i][j] = dsigmoid(hidden.data[i][j]);
      }
    }
    
    hgradient.mMult(hErrors);
    hgradient.mMult(lr);
    
    Matrix inputsT = inputs.transpose(); 
    Matrix weightihD = hgradient.mProduct(inputsT);
    weightsih.mAdd(weightihD);
  
    biash.mAdd(hgradient);        
  }
    
   float sigmoid(float x){

     
     x = exp(x)/(exp(x)+1);
     
  
    return x;
    
    
  }
  
  float dsigmoid(float y){
    return y * (1-y);
  }
  
};
