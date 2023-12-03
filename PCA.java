import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.Set;

import ml.classifiers.TwoLayerNN;
import ml.data.DataSet;
import ml.data.DataSetSplit;
import ml.data.Example;
import ml.data.FeatureNormalizer;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.util.Scanner;


/**
 * Class that implements the PCA algorithm with SVD and eigen vector covariance matrix
 * 
 * @author Joshua Garcia-Kimble, Abraham Arias, Ash Shah
 */
public class PCA {

    public static void main(String[] args) {
        String projectDataFolderPath = "C:\\Users\\Joshua\\Documents\\College\\Senior\\MachineLearning\\final-project\\machine-learning-final-project\\data\\project_data\\";
        String dataTrainFileName = "cs-training.csv";
        String experimentDataFolderPath = "C:\\Users\\Joshua\\Documents\\College\\Senior\\MachineLearning\\final-project\\machine-learning-final-project\\data\\experiment_data\\";
        String experimentFileName = "testing_covariance.csv";
        String dataTrainFilePath = projectDataFolderPath + dataTrainFileName;
        String experimentTrainFilePath = experimentDataFolderPath + experimentFileName;

        DataSet dataTrain = new DataSet(dataTrainFilePath, DataSet.CSVFILE);
        DataSetSplit dataSplit = dataTrain.split(0.1);
        DataSet dataTrainSmall = dataSplit.getTrain();
        PCA pca = new PCA(dataTrainFilePath, 5, PCA_Type.EIGEN);

        // Test out covariance matrix, it should look like:
        // [2.5, 7.5]
        // [0.0, 22.5]
        // DataSet dataTrainExp = new DataSet(experimentTrainFilePath, DataSet.CSVFILE);
        // PCA pcaExp = new PCA(experimentTrainFilePath, 5, PCA_Type.EIGEN);
    }

    /** 
     * Used to specify the type of PCA we want to do.
     * We support the SVD and Eigen Vectors of the Covariance Matrix methods.
     */
    public enum PCA_Type {
        SVD,
        EIGEN
    }

    private static final int NN_HIDDEN_NODES = 20;

    // The type of PCA to do
    private PCA_Type pcaType;

    // The k most vectors we want from the algorithm
    private int k;

    // The data we will do pca on
    private DataSet modifiedData;

    // Get original data used with the NN
    private DataSet originalData;

    private DataSet pcaData;

    /**
     * The constructor assumes the data is in a csv file and that we want the k most important feature combinations
     */
    public PCA(String dataFilePath, int k, PCA_Type pcaType) {
        FeatureNormalizer featureNormalizer = new FeatureNormalizer();
        this.originalData = new DataSet(dataFilePath, DataSet.CSVFILE);
        this.modifiedData = new DataSet(dataFilePath, DataSet.CSVFILE);
        // We normalize the features since this will make our Eigen vector pca calculations better 
        featureNormalizer.preprocessTrain(this.modifiedData);;
        this.k = k;
        this.pcaType = pcaType;
        
        // Populate pcaData with the projected data 
        this.pcaData = this.createPcaData();

        this.getAccuracyOnFullData(this.modifiedData);
        this.getAccuracyOnFullData(this.pcaData);
        System.out.println("===================");
        this.getF1ScoreOnFullData(this.modifiedData);
        this.getF1ScoreOnFullData(this.pcaData);
        // System.out.println(this.originalData.getData());
        // double[] featureMeans = this.getAllFeatureMeans(this.originalData);
        // double[][] covMatrix = this.getCovarianceMatrix(this.originalData, featureMeans);
        // for (double[] row : covMatrix) {
        //     System.out.println(Arrays.toString(row));
        // }
        
    }

    public double getAccuracyOnFullData(DataSet data) {
        // We train the NN on the full data and output the accuracy
        TwoLayerNN nn = new TwoLayerNN(NN_HIDDEN_NODES);
        nn.train(data);
        Double accuracy = nn.classifyData(data);
        System.out.println("Accuracy: " + accuracy);
        return accuracy;
    }

    public double getF1ScoreOnFullData(DataSet data) {
        // We train the NN on the full data and output the accuracy
        TwoLayerNN nn = new TwoLayerNN(NN_HIDDEN_NODES);
        nn.train(data);
        Double f1Score = nn.classifyDataF1Score(data);
        System.out.println("F1Score: " + f1Score);
        return f1Score;
    }

    private DataSet getPcaData() {
        return this.pcaData;
    }

    private DataSet getModifiedData() {
        return this.modifiedData;
    }

    private DataSet getOriginalData() {
        return this.originalData;
    }

    private DataSet createPcaData() {
        // TODO: Calculate the covariance matrix
        double[] featureMeans = this.getAllFeatureMeans(this.modifiedData);
        double[][] covarianceMatrix = this.getCovarianceMatrix(this.modifiedData, featureMeans);

        try {
            double[] eigenValues = PCA.getEigenValues(covarianceMatrix);
            double[][] eigenVectors = PCA.getEigenVectors(covarianceMatrix);
            // Now we take the top k eigenValues and their corresponding eigen vectors
            TopKEigenValuesAndVectors topKEigenValuesAndVectors = this.getTopKEigenValuesAndVectors(k, eigenValues, eigenVectors);
            double[][] topKEigenVectors = topKEigenValuesAndVectors.getEigenVectors();
            // Now, for each example, we project it onto each of the k eigen vectors and create a new example with these data points
            // We then put all of these examples into a new dataset
            return this.runPCA(this.modifiedData, topKEigenVectors);
            // System.out.println("eigenValues: " + Arrays.toString(eigenValues));
            // System.out.println("eigenVectors: " + PCA.make2DArrayString(eigenVectors));

        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
            return null;
        } catch (InterruptedException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
            return null;
        }
    }

    private DataSet runPCA(DataSet data, double[][] eigenVectors) {
        // We create the featureMap for our new dataset
        HashMap<Integer, String> featureMap = new HashMap<>();
        for (int i = 0; i < eigenVectors.length; i++) {
            featureMap.put(i, "eigenVector_" + i);
        }
        // System.out.println("Feature Map");
        // System.out.println(featureMap);
        DataSet pcaOutputData = new DataSet(featureMap);
        
        // Loop through data
        for (Example example : data.getData()) {
            // Assumes that the example has features from 0 to data.getAllFeatureIndices().size()-1 [inclusive]
            // For each example, we create a vector of its features
            double[] featureVector = this.getExampleFeatureVector(example, data.getAllFeatureIndices().size());
            Example pcaExample = new Example();
            pcaExample.setLabel(example.getLabel());
            // We then project that feature vector onto each eigen vector
            for (int j = 0; j < eigenVectors.length; j++) {
                double[] eigenVector = eigenVectors[j];
                double projection = this.getProjection(featureVector, eigenVector);
                pcaExample.setFeature(j, projection);
            }
            // After performing every projection, we make these into a new example and add it to a new dataset 
            pcaOutputData.addData(pcaExample);
        }

        
        return pcaOutputData;
    }

    private double getProjection(double[] featureVector, double[] eigenVector) {
        double projection = 0;
        double eigenVectorMagnitude = 0;
        if (featureVector.length != eigenVector.length) {
            System.out.println("FEATURE VECTORS AND EIGEN VECTORS ARE NOT THE SAME LENGTH");
            return Double.MAX_VALUE;
        }

        for (int i = 0; i < featureVector.length; i++) {
            double featureNum = featureVector[i];
            double eigenNum = eigenVector[i];

            projection += featureNum*eigenNum;
            eigenVectorMagnitude += eigenNum*eigenNum;
        }

        return projection / eigenVectorMagnitude;
    }

    private double[] getExampleFeatureVector(Example example, int numFeatures) {
        double[] featureVector = new double[numFeatures];

        for (int i = 0; i < numFeatures; i++) {
            featureVector[i] = example.getFeature(i);
        }

        return featureVector;
    }

    private TopKEigenValuesAndVectors getTopKEigenValuesAndVectors(int k, double[] eigenValues, double[][] eigenVectors) {
        if (k > eigenValues.length) {
            return null;
        }
        Pair[] eigenValueAndIndex = new Pair[eigenValues.length];
        for (int index = 0; index < eigenValues.length; index++) {
            double val = eigenValues[index];
            Pair pair = new Pair(val, index);
            eigenValueAndIndex[index] = pair;
        }

        Arrays.sort(eigenValueAndIndex);

        double[] topKEigenValues = new double[k];
        double[][] topKEigenVectors = new double[k][eigenVectors[0].length];

        for (int i = 0; i < k; i++) {
            Pair pair = eigenValueAndIndex[i];
            topKEigenValues[i] = pair.getValue();
            topKEigenVectors[i] = eigenVectors[pair.getIndex()];
        }
        TopKEigenValuesAndVectors topKEigenVectorsAndValues = new TopKEigenValuesAndVectors(k, topKEigenValues, topKEigenVectors);
        return topKEigenVectorsAndValues;
    }

    private static String make2DArrayString(double[][] matrix) {
        String ret = "[";
        for (int i = 0; i < matrix.length; i++) {
            if (i != matrix.length-1) {
                ret += Arrays.toString(matrix[i]) + ", ";
            } else {
                ret += Arrays.toString(matrix[i]);
            }
        }
        return ret + "]";
    }

    // This code is an amalgamation of stack overflow, chat-gpt, and our own ingenious
    private static double[] getEigenValues(double[][] matrix) throws IOException, InterruptedException {

        String filename = "EigenValueCalculator.py";
        
        // System.out.println("Matrix: " + PCA.make2DArrayString(matrix));
        String matrixString = PCA.make2DArrayString(matrix);
        // BufferedWriter out = new BufferedWriter(new FileWriter(filename));
        // String command = "python " + filename;// + " < " + matrixString;
        // Process p = Runtime.getRuntime().exec(command);
        ProcessBuilder processBuilder = new ProcessBuilder("python", filename);
        processBuilder.redirectInput(ProcessBuilder.Redirect.PIPE);
        Process process = processBuilder.start();

        // Pass matrixString as input to the Python script
        process.getOutputStream().write(matrixString.getBytes());
        process.getOutputStream().close();

        BufferedReader in = new BufferedReader(new InputStreamReader(process.getInputStream()));//(new InputStreamReader(p.getInputStream()));
        String eigenvaluesString = in.readLine();

        double[] eigenValues = MatrixIO.parseArray(eigenvaluesString);
        return eigenValues;
    }

    // This code is an amalgamation of stack overflow, chat-gpt, and our own ingenious
    private static double[][] getEigenVectors(double[][] matrix) throws IOException, InterruptedException {
        

        String filename = "EigenVectorCalculator.py";
        
        // System.out.println("Matrix: " + PCA.make2DArrayString(matrix));
        String matrixString = PCA.make2DArrayString(matrix);
        // BufferedWriter out = new BufferedWriter(new FileWriter(filename));
        // String command = "python " + filename;// + " < " + matrixString;
        // Process p = Runtime.getRuntime().exec(command);
        ProcessBuilder processBuilder = new ProcessBuilder("python", filename);
        processBuilder.redirectInput(ProcessBuilder.Redirect.PIPE);
        Process process = processBuilder.start();

        // Pass matrixString as input to the Python script
        process.getOutputStream().write(matrixString.getBytes());
        process.getOutputStream().close();

        BufferedReader in = new BufferedReader(new InputStreamReader(process.getInputStream()));//(new InputStreamReader(p.getInputStream()));
        String eigenVectorsString = in.readLine();
        double[][] eigenVectors = MatrixIO.parseMatrix(eigenVectorsString);
        return eigenVectors;
    }

    // This calculation assumes each example has the exact same number of features 
    private double[][] getCovarianceMatrix(DataSet data, double[] featureMeans) {
        int numExamples = data.getData().size();
        // First we make a numFeatures x numFeatures matrix
        int numFeatures = data.getAllFeatureIndices().size();
        double[][] covarianceMatrix = new double[numFeatures][numFeatures];
        // We go through each example and then each feature in that example in a nested way to get the variance

        // This for loop assumes that each example contains all of the features in 
        for (Example example : data.getData()) {
            Set<Integer> featureSet = example.getFeatureSet();
            for (int feature1 = 0 ; feature1 < numFeatures; feature1++) {
                for (int feature2 = feature1 ; feature2 < numFeatures; feature2++) {
                    double feature1MeanDifference = example.getFeature(feature1) - featureMeans[feature1];
                    double feature2MeanDifference = example.getFeature(feature2) - featureMeans[feature2];
                    double almostVarTerm = feature1MeanDifference*feature2MeanDifference;
                    covarianceMatrix[feature1][feature2] += almostVarTerm;
                    covarianceMatrix[feature2][feature1] += almostVarTerm;
                }
            }
        }

        // Now we must devide by N-1
        for (int feature1 = 0 ; feature1 < numFeatures; feature1++) {
            for (int feature2 = feature1 ; feature2 < numFeatures; feature2++) {
                covarianceMatrix[feature1][feature2] = covarianceMatrix[feature1][feature2]/(numExamples-1);
                if (feature1 != feature2) {
                    covarianceMatrix[feature2][feature1] = covarianceMatrix[feature2][feature1]/(numExamples-1);
                }
            }
        }
        return covarianceMatrix;
            // The nested for loop will have us calculating the covariance of each variable (including itself)

        // Loop through each example
            // Calculate x_i - mean_x for all features x
            // Multiple them together
    }

    private double[] getAllFeatureMeans(DataSet data) {
        
        int numFeatures = data.getAllFeatureIndices().size();
        double[] featureMeans = new double[numFeatures];
        int[] featureCounts = new int[numFeatures];

        for (Example example : data.getData()) {
            for (Integer feature : example.getFeatureSet()) {
                featureMeans[feature] += example.getFeature(feature);
                featureCounts[feature] += 1;
            }
        }

        for (int feature = 0; feature < numFeatures; feature++) {
            featureMeans[feature] /= featureCounts[feature];
        }
        return featureMeans;
    }

    private class MatrixIO {
        public static double[] parseArray(String arrayString) {
            String[] elements = arrayString.split(" ");
            double[] result = new double[elements.length];
            for (int i = 0; i < elements.length; i++) {
                String s = elements[i];
                s = s.replace("[", "");
                s = s.replace("]", "");
                s = s.replace(",", "");
                result[i] = Double.parseDouble(s);
            }
            return result;
        }
    
        public static double[][] parseMatrix(String matrixString) {
            String[] s = matrixString.split("],");
            double[][] matrix = new double[s.length][s.length];
            for (int i = 0; i < s.length; i++) {
                String str = s[i].trim();
                double[] row = parseArray(str);
                matrix[i] = row;
            }
            return matrix;
        }
    }


}