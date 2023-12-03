import java.util.Arrays;
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
        // pca.getAccuracyOnFullData(dataTrainSmall);

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

        if (this.pcaType == PCA_Type.EIGEN) {

        } else if (this.pcaType == PCA_Type.SVD) {

        }
        // System.out.println(this.originalData.getData());
        // double[] featureMeans = this.getAllFeatureMeans(this.originalData);
        // double[][] covMatrix = this.getCovarianceMatrix(this.originalData, featureMeans);
        // for (double[] row : covMatrix) {
        //     System.out.println(Arrays.toString(row));
        // }

        this.eigenPca();
        /**
         * TODO: Run PCA in the constructor. I think we should make one function for SVD PCA and one function for Eigen covariance matrix PCA 
         * then just call those from here. 
         */

         // Calculate covariance matrix

         // Extract eigen vectors from covariate matrix

         // Get eigen values corresponding to eigen vectors

         // Rank eigen vectors by how much variance they explained

         // Pick the top k eigen vectors

         // Project data onto these top k eigen vectors
        
    }

    public double getAccuracyOnFullData(DataSet data) {
        // We train the NN on the full data and output the accuracy
        TwoLayerNN nn = new TwoLayerNN(NN_HIDDEN_NODES);
        nn.train(data);
        Double accuracy = nn.classifyData(data);
        System.out.println("Accuracy: " + accuracy);
        return accuracy;
    }

    private void eigenPca() {
        // TODO: Calculate the covariance matrix
        double[] featureMeans = this.getAllFeatureMeans(this.modifiedData);
        double[][] covarianceMatrix = this.getCovarianceMatrix(this.modifiedData, featureMeans);

        try {
            double[] eigenValues = PCA.getEigenValues(covarianceMatrix);
            double[][] eigenVectors = PCA.getEigenVectors(covarianceMatrix);
            System.out.println("eigenValues: " + Arrays.toString(eigenValues));
            System.out.println("eigenVectors: " + PCA.make2DArrayString(eigenVectors));
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        } catch (InterruptedException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
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
        // Process process = new ProcessBuilder("python", "EigenValueCalculator.py")
        //         .start();
        // try (BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(process.getOutputStream()))) {
        //     MatrixIO.writeMatrix(writer, matrix);
        // }

        // BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
        // String eigenvaluesString = reader.readLine();
        // process.waitFor();

        // return MatrixIO.parseArray(eigenvaluesString);
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

        // Process process = new ProcessBuilder("python", "EigenVectorCalculator.py")
        //         .start();

        // try (BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(process.getOutputStream()))) {
        //     MatrixIO.writeMatrix(writer, matrix);
        // }

        // BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
        // String eigenvectorsString = reader.readLine();
        // process.waitFor();

        // return MatrixIO.parseMatrix(eigenvectorsString);
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


    private void Svd_Pca() {

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