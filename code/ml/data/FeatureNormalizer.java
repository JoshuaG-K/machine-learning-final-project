// Joshua Garcia-Kimbe
package ml.data;

import java.util.ArrayList;
import java.util.HashMap;

/**
 * Class used to normalize a dataset by feature (column)
 */
public class FeatureNormalizer implements DataPreprocessor {
    // Get the mean and variance of each feature
    private HashMap<Integer, Double> trainMeanPerFeature;
    private HashMap<Integer, Double> trainVariancePerFeature;

    public static void main(String[] args) {
        String folderPath = "C:\\Users\\Joshua\\Documents\\College\\Senior\\MachineLearning\\assign4\\assignment-4-joshuagk\\data\\";
        String fakeTitanicFile = "titanic-train.csv";
        String realTitanicFile = "titanic-train.real.csv";

        DataSet fakeTitanicData = new DataSet(folderPath + fakeTitanicFile, DataSet.CSVFILE);
        DataSet realTitanicData = new DataSet(folderPath + realTitanicFile, DataSet.CSVFILE);

        DataSetSplit dataSplit = realTitanicData.split(0.8);

        FeatureNormalizer featureNormalizer = new FeatureNormalizer();

        ArrayList<Example> pretrainData = dataSplit.getTrain().getData();

        for(int i = 0; i < 10; i++) {
            Example example = pretrainData.get(i);
        }
        featureNormalizer.preprocessTrain(dataSplit.getTrain());
        for(int i = 0; i < 10; i++) {
            Example example = pretrainData.get(i);
            // System.out.println("Example: " + example);
        }
    }

    /**
     * Constructed, used to initialize the two variables in this class
     */
    public FeatureNormalizer() {
        trainMeanPerFeature = new HashMap<>();
        trainVariancePerFeature = new HashMap<>();
    }

    /**
     * Preprocesses the training data by normalizing each feature. Calculates
     * the mean and variance and uses that to normalize all features
     * @param train , dataset to normalize and calculate mean and variances from
     */
    @Override
    public void preprocessTrain(DataSet train) {
        ArrayList<Example> data = train.getData();
        Integer n = data.size();
        // Get feature average
        // Loop through each example to add up all feature values per feature
        for (Example example : data) {
            for (Integer feature : example.getFeatureSet()) {
                Double currentMeanValue = trainMeanPerFeature.getOrDefault(feature, 0.0);
                Double currentFeatureValue = example.getFeature(feature);
                trainMeanPerFeature.put(feature, currentMeanValue + currentFeatureValue);
            }
        }

        // Divide by n for all features to get average
        for (Integer feature : trainMeanPerFeature.keySet()) {
            Double currentMeanValue = trainMeanPerFeature.getOrDefault(feature, 0.0);
            trainMeanPerFeature.put(feature, currentMeanValue/n);
        }
        
        // Now calculate the variance for each feature

        // Loop through each example to add up all (feature values minus mean)
        for (int i = 0; i < data.size(); i++) {
            Example example = data.get(i);
            for (Integer feature : example.getFeatureSet()) {
                // Get the current aggregate variance value
                Double currentVarianceValue = trainVariancePerFeature.getOrDefault(feature, 0.0);
                if (currentVarianceValue.isInfinite()) {
                    trainVariancePerFeature.put(feature, 0.0);
                    currentVarianceValue = 0.0;
                }
                // Get the value for the feature
                Double currentFeatureValue = example.getFeature(feature);
                // Get the mean for the feature
                Double meanOfFeature = trainMeanPerFeature.getOrDefault(feature, 0.0);
                // Add (mean - value) to current aggregate variance value
                Double valueToAdd = Math.pow((meanOfFeature - currentFeatureValue), 2);
                Double newCurrentVarianceValue = currentVarianceValue + Math.pow((meanOfFeature - currentFeatureValue), 2);
                trainVariancePerFeature.put(feature, newCurrentVarianceValue);
            }
        }
        
        // Divide by n for all features to get average
        for (Integer feature : trainVariancePerFeature.keySet()) {
            Double currentVarianceValue = trainVariancePerFeature.getOrDefault(feature, 0.0);
            trainVariancePerFeature.put(feature, Math.pow(currentVarianceValue/(n-1), 0.5));
        }

        // Now we actually alter the data

        // Go through all the examples in the data set

        // For each example, recenter and variance scale

        for (Example example : train.getData()) {
            for (Integer feature : example.getFeatureSet()) {
                Double currentFeatureValue = example.getFeature(feature);
                Double mean = trainMeanPerFeature.get(feature);
                Double std = trainVariancePerFeature.get(feature);
                Double scaledFeatureValue = (currentFeatureValue - mean)/std;
                example.addFeature(feature, scaledFeatureValue);
            }
        }
    }

    /**
     * Preprocesses the training data by normalizing each feature. Uses the 
     * mean and variance from the train method to normalize all features
     * @param test , dataset to normalize and calculate mean and varainces from
     */
    @Override
    public void preprocessTest(DataSet test) {
        if (this.trainMeanPerFeature == null || this.trainVariancePerFeature == null) {
            return;
        }
         for (Example example : test.getData()) {
            for (Integer feature : example.getFeatureSet()) {
                Double currentFeatureValue = example.getFeature(feature);
                Double mean = trainMeanPerFeature.get(feature);
                Double variance = trainVariancePerFeature.get(feature);
                Double scaledFeatureValue = (currentFeatureValue - mean)/variance;
                example.addFeature(feature, scaledFeatureValue);
            }
        }
    }
    
}
