// Joshua Garcia-Kimble
package ml.classifiers;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;

import ml.data.CrossValidationSet;
import ml.data.DataSet;
import ml.data.DataSetSplit;
import ml.data.Example;

/**
 * Class that implements a TwoLayerNN (one output layer and one hidden layer) for classification with a confidence
 * 
 * @author Joshua Garcia-Kimble
 */
public class TwoLayerNN implements Classifier {

    public static void main(String[] args) {
        String dataFolder = "C:\\Users\\Joshua\\Documents\\College\\Senior\\MachineLearning\\assign8\\assignment-8-joshua8\\data\\";
        String testFilename = "test-example.csv";
        String titanicFilename = "titanic-train.csv";
        String pathToTestFile = dataFolder + testFilename;
        String pathToTitanicFile = dataFolder + titanicFilename;

        DataSet testData = new DataSet(pathToTestFile, DataSet.CSVFILE);
        DataSet titanicData = new DataSet(pathToTitanicFile, DataSet.CSVFILE);
        DataSetSplit titanicSplit = titanicData.split(0.9);

        // Question 4
        TwoLayerNN.findBestModel(titanicData);

        // Question 3
        // TwoLayerNN.question3(titanicData);

        // Question 2

        // TwoLayerNN nn = new TwoLayerNN(10);
        // // nn.setIterations(100);
        // nn.trainAndTrack(titanicSplit.getTest(), titanicSplit.getTrain());

        // Question 1

        // System.out.println("Test Data: " +  testData.getData());
        // System.out.println("Test Data Feature Size: " + testData.getAllFeatureIndices().size());

        // int numHiddenNodes = 2;
        // TwoLayerNN nn = new TwoLayerNN(numHiddenNodes);
        // nn.setEta(0.5);
        // nn.train(testData);

        // double[][] firstWeights = nn.getFirstWeights();
        // System.out.println("========== WEIGHTS ==========");
        // for (int i = 0; i < firstWeights.length; i++) {
        //     System.out.println("firstWeights " + i + " : " + Arrays.toString(firstWeights[i]));
        // }
        // double[] secondWeights = nn.getSecondWeights();
        // System.out.println("Second weights: " + Arrays.toString(secondWeights));

        // double[] hiddenLayerActivations = nn.getHiddenLayerActivations();
        // double outputActivation = nn.getOutputActivation();
        // double outputPreActivation = nn.getoutputPreActivation();
        // System.out.println("========= Activations ============");
        // System.out.println("hiddenLayerActivations: " + Arrays.toString(hiddenLayerActivations));
        // System.out.println("Output activation: " + outputActivation);
        // System.out.println("Output pre-activation: " + outputPreActivation);
        
        
        // double xVal = 5.0;
        // double derivative = nn.activationDerivative(xVal);
        // System.out.print("Derivative at " + xVal + ": " + derivative);
        
    }

    // Have a weight matrix where each row represents the node. That means that
    // Each row will have weight vectors of size d where d is the dimensions of the input 
    // which is akin to all of the input's features going to one node. We do this for
    // numHiddenNodes (rows in the weight matrix) and it works. 
    // Length of weight feature vector will be the number of features in the dataset.
    private int numHiddenNodes;
    private double eta;
    private int numIterations;
    private int numFirstWeights;
    private int numSecondWeights;

    private double[][] firstWeights;
    private double[][] firstWeightAccumulators;
    private double[][] gradFirstWeights;

    private double[] hiddenLayer;
    private double outputPreActivation;

    private double[] secondWeights;
    private double[] secondWeightAccumulators;
    private double[] gradSecondWeights;

    private double[] lossArray;
    private double[] trainingAccuracyArray;
    private double trainingAccuracyAverage;
    private double[] testingAccuracyArray;
    private double testingAccuracyAverage;

    private Random rd;
    private Set<Integer> currentFeatureSet;

    private DataSet data;
    private int batchSize;

    /**
     * Constructor, takes in the number of hidden nodes
     * @param numHiddenNodes number of hidden nodes
     */
    public TwoLayerNN(int numHiddenNodes) {
        this.numHiddenNodes = numHiddenNodes;
        this.eta = 0.1;
        this.numIterations = 200;
        this.rd = new Random();
    }

    @Override
    public void train(DataSet dataWithNoBias) {
        // Make new dataset have added bias feature
        this.data = dataWithNoBias.getCopyWithBias();
        this.batchSize = this.data.getData().size();

        this.currentFeatureSet = this.data.getAllFeatureIndices();
        // Don't need to add +1 for the bias because this.data has the bias
        this.numFirstWeights = this.currentFeatureSet.size();
        // For this one you still need to tho because it doesn't depend on this.data
        this.numSecondWeights = this.numHiddenNodes+1;
        // Set up hidden layer
        this.hiddenLayer = new double[this.numHiddenNodes+1];
        // Set the bias of the hidden layer to 1
        this.hiddenLayer[this.hiddenLayer.length-1] = 1.0;
        // Set up weights
        this.firstWeights = new double[this.numHiddenNodes][this.numFirstWeights];
        this.gradFirstWeights = new double[this.numHiddenNodes][this.numFirstWeights];
        this.firstWeightAccumulators = new double[this.numHiddenNodes][this.numFirstWeights];

        this.secondWeights = new double[this.numSecondWeights];
        this.gradSecondWeights = new double[this.numSecondWeights];
        this.secondWeightAccumulators = new double[this.numSecondWeights]; 

        this.initializeWeights();

        boolean usingStartingNetwork = false;

        if (this.numHiddenNodes == 2 && this.data.getAllFeatureIndices().size() == 3) {
            System.out.println("UTILIZING STARTING NETWORK");
            this.makeIntoStartingNetwork();
            usingStartingNetwork = true;
        }

        // Set up loss list to keep track
        lossArray = new double[numIterations];

        for (int iter = 0; iter < this.numIterations; iter++) {
            // Randomly shuffle the data
            ArrayList<Example> exampleList = data.getData();
            Collections.shuffle(exampleList);
            double sumOfLoss = 0.0;
            // System.out.println("Example list size: " + exampleList.size());
            for (Example example : exampleList) {
                // Calculate the final output by doing "matrix" multiplication - make this into a function
                // double prediction = this.forward(example);
                
                double loss = this.calculateLoss(example);
                sumOfLoss += loss;    
                // Back propogate the error to the weights - make this into a function
                this.updateWeightAccumulators(example);
            }
            this.updateWeights();
            this.resetWeightAccumulators();
            double averageLoss = sumOfLoss / data.getData().size();
            lossArray[iter] = sumOfLoss;
            
            if (usingStartingNetwork) {
                break;
            }
        }
    }

    /**
     * Trains the model with the train data and tracks the accuracy of the training data and testing data
     * @param dataTrainWithNoBias training data for the model during this training cycle
     * @param dataTestWithNoBias testing data for the model during this testing cycle
     */
    public void trainAndTrack(DataSet dataTrainWithNoBias, DataSet dataTestWithNoBias) {
        // Make new dataset have added bias feature
        DataSet testData = dataTestWithNoBias.getCopyWithBias();
        DataSet trainData = dataTrainWithNoBias.getCopyWithBias();
        this.batchSize = trainData.getData().size();
        // System.out.println("Train data size: " + trainData.getData().size());
        // System.out.println("Test data size: " + testData.getData().size());

        this.currentFeatureSet = trainData.getAllFeatureIndices();
        // Don't need to add +1 for the bias because trainData has the bias
        this.numFirstWeights = this.currentFeatureSet.size();
        // For this one you still need to tho because it doesn't depend on trainData
        this.numSecondWeights = this.numHiddenNodes+1;
        // Set up hidden layer
        this.hiddenLayer = new double[this.numHiddenNodes+1];
        // Set the bias of the hidden layer to 1
        this.hiddenLayer[this.hiddenLayer.length-1] = 1.0;
        // Set up weights
        this.firstWeights = new double[this.numHiddenNodes][this.numFirstWeights];
        this.gradFirstWeights = new double[this.numHiddenNodes][this.numFirstWeights];
        this.firstWeightAccumulators = new double[this.numHiddenNodes][this.numFirstWeights];

        this.secondWeights = new double[this.numSecondWeights];
        this.gradSecondWeights = new double[this.numSecondWeights];
        this.secondWeightAccumulators = new double[this.numSecondWeights]; 

        this.initializeWeights();

        // Done if we are using the example network that Doctor Dave gave us
        boolean usingStartingNetwork = false;

        if (this.numHiddenNodes == 2 && trainData.getAllFeatureIndices().size() == 3) {
            System.out.println("UTILIZING STARTING NETWORK");
            this.makeIntoStartingNetwork();
            usingStartingNetwork = true;
        }

        // Set up loss list to keep track
        lossArray = new double[numIterations];
        trainingAccuracyArray = new double[numIterations];
        testingAccuracyArray = new double[numIterations];
        trainingAccuracyAverage = 0.0;
        testingAccuracyAverage = 0.0;

        // Actually train the model
        for (int iter = 0; iter < this.numIterations; iter++) {
            // Randomly shuffle the data
            ArrayList<Example> exampleList = trainData.getData();
            Collections.shuffle(exampleList);
            double sumOfLoss = 0.0;
            
            // Update weight accumulators for each example
            for (Example example : exampleList) {
                // Calculate the final output by doing "matrix" multiplication - make this into a function
                // double prediction = this.forward(example);
                
                double loss = this.calculateLoss(example);
                sumOfLoss += loss;    
                // Back propogate the error to the weights - make this into a function
                this.updateWeightAccumulators(example);
            }
            // Update the actual weights
            this.updateWeights();
            this.resetWeightAccumulators();
            double averageLoss = sumOfLoss / trainData.getData().size();

            lossArray[iter] = sumOfLoss;
            double trainingAccuray = this.classifyData(trainData);
            trainingAccuracyAverage+=trainingAccuray;
            double testingAccuracy = this.classifyData(testData);
            testingAccuracyAverage+=testingAccuracy;

            trainingAccuracyArray[iter] = trainingAccuray;
            testingAccuracyArray[iter] = testingAccuracy;

            if (usingStartingNetwork) {
                break;
            }
        }
        trainingAccuracyAverage/=numIterations;
        testingAccuracyAverage/=numIterations;
        // System.out.println("Sum of loss: " + Arrays.toString(lossArray));
        // System.out.println("Training Accuracy: " + Arrays.toString(trainingAccuracyArray));
        // System.out.println("Testing Accuracy: " + Arrays.toString(testingAccuracyArray));
    }

    /**
     * Finds the best eta value for a model
     * @param dataWithNoBias the data that will be used to train and test
     */
    public static void findBestModel(DataSet dataWithNoBias) {
        DataSet q3Data = dataWithNoBias.getCopyWithBias();
        CrossValidationSet cvSet = new CrossValidationSet(q3Data, 10);


        // DataSetSplit q3DataSplit = q3Data.split(0.9);
        double minEta = 0.05;
        double maxEta = 1;
        double increment = 0.05;

        double[] trainingAccuracyForNumHiddenNodes = new double[(int)Math.floor((maxEta-minEta)/increment)+1];
        double[] testingAccuracyForNumHiddenNodes = new double[(int)Math.floor((maxEta-minEta)/increment)+1];
        double[] etaArray = new double[(int)Math.floor((maxEta-minEta)/increment)+1];

        int count = 0;
        int bestNumHiddenNodes = 8;

        // FInd the best eta value 
        for (double eta = minEta; eta <= maxEta; eta+=increment) {
            double averageTrainingAccuracy = 0.0;
            double averageTestingAccuracy = 0.0;
            // Go through each cv split and get average accuracies
            for (int i = 0; i < cvSet.getNumSplits(); i++) {
                TwoLayerNN nn = new TwoLayerNN(bestNumHiddenNodes);
                nn.setEta(eta);
                DataSetSplit q3DataSplit = cvSet.getValidationSet(i);
                nn.trainAndTrack(q3DataSplit.getTrain(), q3DataSplit.getTest());
                // System.out.println("Training Accuracy: " + trainingAccuracy);
                averageTrainingAccuracy += nn.getTrainingAccuracyAverage();;
                averageTestingAccuracy += nn.getTestingAccuracyAverage();;
            }
            trainingAccuracyForNumHiddenNodes[count] = averageTrainingAccuracy/cvSet.getNumSplits();
            testingAccuracyForNumHiddenNodes[count] = averageTestingAccuracy/cvSet.getNumSplits();
            etaArray[count] = eta;
            count++;
        }
        
        System.out.println("Training Accuracy Per Eta: " + Arrays.toString(trainingAccuracyForNumHiddenNodes));
        System.out.println("Testing Accuracy Per Eta: " + Arrays.toString(testingAccuracyForNumHiddenNodes));
        System.out.println("Eta: " + Arrays.toString(etaArray));
    }

    /**
     * Used to get the training and testing accuracies for each NN with a different number of hidden Nodes
     * @param dataWithNoBias
     */
    public static void question3(DataSet dataWithNoBias) {
        DataSet q3Data = dataWithNoBias.getCopyWithBias();
        CrossValidationSet cvSet = new CrossValidationSet(q3Data, 10);


        // DataSetSplit q3DataSplit = q3Data.split(0.9);
        int minNumNodes = 1;
        int maxNumNodes = 10;
        int increment = 1;

        double[] trainingAccuracyForNumHiddenNodes = new double[(int)Math.floor((maxNumNodes-minNumNodes)/increment)+1];
        double[] testingAccuracyForNumHiddenNodes = new double[(int)Math.floor((maxNumNodes-minNumNodes)/increment)+1];

        int count = 0;

        for (int numHiddenNodes = minNumNodes; numHiddenNodes <= maxNumNodes; numHiddenNodes+=increment) {
            double averageTrainingAccuracy = 0.0;
            double averageTestingAccuracy = 0.0;
            for (int i = 0; i < cvSet.getNumSplits(); i++) {
                TwoLayerNN nn = new TwoLayerNN(numHiddenNodes);
                DataSetSplit q3DataSplit = cvSet.getValidationSet(i);
                nn.trainAndTrack(q3DataSplit.getTrain(), q3DataSplit.getTest());
                // System.out.println("Training Accuracy: " + trainingAccuracy);
                averageTrainingAccuracy += nn.getTrainingAccuracyAverage();;
                averageTestingAccuracy += nn.getTestingAccuracyAverage();;
            }
            trainingAccuracyForNumHiddenNodes[count] = averageTrainingAccuracy/cvSet.getNumSplits();
            testingAccuracyForNumHiddenNodes[count] = averageTestingAccuracy/cvSet.getNumSplits();
            count++;
        }
        
        System.out.println("Training Accuracy Per Number of Hidden Nodes: " + Arrays.toString(trainingAccuracyForNumHiddenNodes));
        System.out.println("Testing Accuracy Per Number of Hidden Nodes: " + Arrays.toString(testingAccuracyForNumHiddenNodes));
    }

    @Override
    public double classify(Example example) {
        double prediction = this.forward(example);
        return prediction > 0 ? 1.0 : -1.0 ;
    }

    /**
     * Returns the average accuracy of this classifier on each example in the dataset
     * @param test the data set
     * @return average accuracy
     */
    public double classifyData(DataSet test) {
		Double accuracy = 0.0;
		for (Example example : test.getData()) {
			if (example.getLabel() == this.classify(example)) {
				accuracy += 1.0;
			}
		}
		return accuracy / test.getData().size();
	}

    @Override
    public double confidence(Example example) {
        return this.forward(example);    
    }

    /**
     * Sets the eta value
     * @param newEta the new eta value
     */
    public void setEta(double newEta) {
        this.eta = newEta;
    }

    /**
     * Sets the number of iterations
     * @param newNumIterations
     */
    public void setIterations(int newNumIterations) {
        this.numIterations = newNumIterations;
    }

    /**
     * Initializes the weights of the model to be 0
     */
    private void initializeWeights() {
        // Initialize weights randomly for firstWeights
        for (int i = 0; i < firstWeights.length; i++) {
            for (int j = 0; j < firstWeights[0].length; j++) {
                firstWeights[i][j] = this.getRandomWeight();
            }
        }

        // Initialize weights randomly for second weights
        for (int i = 0; i < secondWeights.length; i++) {
            secondWeights[i] = this.getRandomWeight();
        }
    }

    /**
     * Gets a random weight centered between -0.1 and 0.1
     * @return
     */
    private double getRandomWeight() {
        // Gets uniform value between 0 and 1
        double weightInit = rd.nextDouble();
        // Make weightInit between -0.1 and 0.1
        weightInit*=2;
        weightInit-=1;
        weightInit/=10;
        return weightInit;
    }

    /**
     * Makes our NN into the starting network that Doctor Dave gave us
     */
    private void makeIntoStartingNetwork() {
        this.firstWeights[0][0] = -0.7;
        this.firstWeights[0][1] = 1.6;
        this.firstWeights[0][2] = -1.8;
        this.firstWeights[1][0] = 0.03;
        this.firstWeights[1][1] = 0.6;
        this.firstWeights[1][2] = -1.4;

        this.secondWeights[0] = -1.1;
        this.secondWeights[1] = -0.6;
        this.secondWeights[2] = 1.8;
    }

    /**
     * Runs an example through one pass of the neural network
     * @param example the example to pass through
     * @return the final output value 
     */
    private double forward(Example example) {
        // No need to add bais to example because we only use a dataset with all the bias added
        // Get all of the features in the example (includes the bias feature)
        Set<Integer> exampleFeatureSet = example.getFeatureSet();

        // Create a hidden layer to temporarily store the outputs 
        // double[] currHiddenLayer = new double[this.numHiddenNodes];

        // Go through the first set of weights
        double output = 0.0;
        double activationOutput = 0.0;
        for (int nodeNumber = 0; nodeNumber < this.firstWeights.length; nodeNumber++) {
            output = 0.0;
            for (Integer feature : exampleFeatureSet) {
                output += example.getFeature(feature)*this.firstWeights[nodeNumber][feature];
            }
            activationOutput = this.activationFunction(output);
            this.hiddenLayer[nodeNumber] = activationOutput;
        }

        // Go through the last set of weights
        output = 0.0;
        for (int nodeNumber = 0; nodeNumber < this.hiddenLayer.length; nodeNumber++) {
            output += this.hiddenLayer[nodeNumber]*this.secondWeights[nodeNumber];
        }

        // // Now we have to account for the bias
        // output += this.secondWeights[this.secondWeights.length-1];
        this.outputPreActivation = output;
        activationOutput = this.activationFunction(output);


        return activationOutput;
    }

    /**
     * Calculates the loss for a given training example
     * @param example training example to calculate loss for 
     * @return the loss
     */
    public double calculateLoss(Example example) {
        double y = example.getLabel();
        double yPrime = this.forward(example);
        return Math.pow(y-yPrime, 2);
    }

    /**
     * Updates the gradient list for both firstWeights and secondWeights. This
     * function uses the values in firstWeights and secondWeights
     * @param example
     */
    private void updateWeightAccumulators(Example example) {
        double prediction = this.forward(example);
        double vDotH = this.outputPreActivation;
        double lastLayerDerivative = this.activationDerivative(vDotH);

        // First, we go through each weight in secondWeights
        for (int i = 0; i < this.secondWeights.length; i++) {
            double y = example.getLabel();
        
            double h_k = this.hiddenLayer[i];
            double gradSecondWeight = (this.eta*h_k*lastLayerDerivative*(y-prediction));
            // System.out.println("Second weight derivative: " + lastLayerDerivative);
            // this.gradSecondWeights[i] += (this.eta*h_k*derivative*(y-prediction));
            this.secondWeightAccumulators[i] += (this.secondWeights[i] + gradSecondWeight);
        }

        // Update the first weight accumulators
        for (int i = 0; i < this.firstWeights.length; i++) {
            // System.out.println("First weights " + i + " : " + Arrays.toString(this.firstWeights[i]));
            double w_kDotProduct = this.getDotProduct(this.firstWeights[i], example);
            // System.out.println("Example " + i + " with dot product:" + w_kDotProduct);
            double w_kDerivative = this.activationDerivative(w_kDotProduct);
            // System.out.println("Has Derivative: " + w_kDerivative);
            for (Integer feature : example.getFeatureSet()) {
                double x_j = example.getFeature(feature.intValue());
                double y = example.getLabel();

                double v_k = this.secondWeights[i];
                double gradFirstWeight = this.eta*x_j*w_kDerivative*v_k*lastLayerDerivative*(y-prediction);
                this.firstWeightAccumulators[i][feature] += (this.firstWeights[i][feature] + gradFirstWeight);
            }
        }
    }

    /**
     * Applies the weight accumulators to the weights 
     */
    private void updateWeights() {
        // Update the second weights from weight accumulators
        for (int i = 0; i < this.secondWeights.length; i++) {
            this.secondWeights[i] = this.secondWeightAccumulators[i]/this.batchSize;
        }

        // Update Gradient for first weights
        for (int i = 0; i < this.firstWeights.length; i++) {
            for (int j = 0; j < this.firstWeights[0].length; j++) {
                this.firstWeights[i][j] = this.firstWeightAccumulators[i][j]/this.batchSize;
            }
        }
    }

    /**
     * Resets the weight accumulators to zero
     */
    private void resetWeightAccumulators() {
        // Update the second weights from weight accumulators
        for (int i = 0; i < this.secondWeightAccumulators.length; i++) {
            this.secondWeightAccumulators[i] = 0.0;
        }

        // Update Gradient for first weights
        for (int i = 0; i < this.firstWeightAccumulators.length; i++) {
            for (int j = 0; j < this.firstWeightAccumulators[0].length; j++) {
                this.firstWeightAccumulators[i][j] = 0.0;
            }
        }

    }

    /**
     * Gets the dot product between the features in example and the corresponding indices for that 
     * feature in array
     * @param array
     * @param example
     * @return the dot product of array and example. 
     */
    public double getDotProduct(double[] array, Example example) {
        double output = 0.0;
        for (Integer feature : example.getFeatureSet()) {
            output += array[feature]*example.getFeature(feature);
        }
        // We don't have to account for the bias because it's already in the example
        return output;
    } 

    /**
     * Assuemes array1 and array2 have the same length, if not returns -1.0
     * @param array1
     * @param array2
     * @return
     */
    public double getDotProduct(double[] array1, double[] array2) {
        if (array1.length != array2.length) {
            return 0.42424242;
        }

        double output = 0.0;
        for (int i = 0; i < array1.length; i++) {
            output += array1[i]*array2[i];
        }
        
        return output;
    } 

    /**
     * The activation function to use
     * @param input
     * @return
     */
    private double activationFunction(double input) {
        return Math.tanh(input);
    }

    /**
     * Returns the derivative of the activation function that we are using. We 
     * are using tanh so we return sech^2
     * @param x
     * @return
     */
    private double activationDerivative(double x) {
        double derivative = 1.0-Math.pow(Math.tanh(x), 2);
        return derivative;
    }

    /**
     * Get the array for the second set of weights
     * @return 
     */
    public double[] getSecondWeights() {
        return this.secondWeights;
    }

    /**
     * Get the array for the first set of weights
     * @return 
     */
    public double[][] getFirstWeights() {
        return this.firstWeights;
    }

    /**
     * Get the array for the hidden layer of activations after running forward
     * @return
     */
    public double[] getHiddenLayerActivations() {
        return this.hiddenLayer;
    }

    /**
     * Gets the final output of the Neural network
     * @return
     */
    public double getOutputActivation() {
        return this.activationFunction(this.outputPreActivation);
    }

    /**
     * This value is created every time .forward is called
     * @return
     */
    private double getOutputPreActivation() {
        return this.outputPreActivation;
    }

    /**
     * This array is created every time .trainAndTrack is called
     * @return
     */
    private double[] getTrainingAccuracyArray() {
        return this.trainingAccuracyArray;
    }

    /**
     * This array is created every time .trainAndTrack is called
     * @return
     */
    private double[] getTestingAccuracyArray() {
        return this.trainingAccuracyArray;
    }

    /**
     * This array is created every time .trainAndTrack and .train is called
     * @return
     */
    private double[] getLossArray() {
        return this.lossArray;
    }

    /**
     * This value is created every time .trainAndTrack is called
     * @return
     */
    public double getTrainingAccuracyAverage() {
        return this.trainingAccuracyAverage;
    }

    /**
     * This value is created every time .trainAndTrack is called
     * @return
     */
    public double getTestingAccuracyAverage() {
        return this.testingAccuracyAverage;
    }

}