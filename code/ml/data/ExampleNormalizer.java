// Joshua Garcia-Kimble
package ml.data;

import java.util.ArrayList;

/**
 * Class used to normalize a dataset by example (row)
 */
public class ExampleNormalizer implements DataPreprocessor {

    public static void main(String[] args) {
        String folderPath = "C:\\Users\\Joshua\\Documents\\College\\Senior\\MachineLearning\\assign4\\assignment-4-joshuagk\\data\\";
        String fakeTitanicFile = "titanic-train.csv";
        String realTitanicFile = "titanic-train.real.csv";

        DataSet fakeTitanicData = new DataSet(folderPath + fakeTitanicFile, DataSet.CSVFILE);
        DataSet realTitanicData = new DataSet(folderPath + fakeTitanicFile, DataSet.CSVFILE);

        DataSetSplit dataSplit = realTitanicData.split(0.8);

        ExampleNormalizer exampleNormalizer = new ExampleNormalizer();
        FeatureNormalizer featureNormalizer = new FeatureNormalizer();

        ArrayList<Example> pretrainData = dataSplit.getTrain().getData();
        System.out.println("=========Pre-trainData==========");
        for(int i = 0; i < 10; i++) {
            Example example = pretrainData.get(i);
            System.out.println("Example: " + example);
        }

        featureNormalizer.preprocessTrain(dataSplit.getTrain());
        System.out.println("=========Post-featureNormalizer-trainData==========");
        for(int i = 0; i < 10; i++) {
            Example example = pretrainData.get(i);
            System.out.println("Example: " + example);
        }

        exampleNormalizer.preprocessTrain(dataSplit.getTrain());
        System.out.println("=========Post-feature&exampleNormalizer-trainData==========");
        for(int i = 0; i < 10; i++) {
            Example example = pretrainData.get(i);
            System.out.println("Example: " + example);
        }

        
    }

    /**
     * Preprocesses the DataSet train by going through each example then going through each feature
     * in the example and normalizing it so that each feature length is 1
     * @param train, the dataset to to normalize
     */
    @Override
    public void preprocessTrain(DataSet train) {
        // Go through all the examples in the data set

        // For each example, normalize

        for (Example example : train.getData()) {
            // Record total length
            Double totalLength = 0.0;
            for (Integer feature : example.getFeatureSet()) {
                totalLength += Math.pow(example.getFeature(feature), 2);
            }
            totalLength = Math.pow(totalLength, 0.5);
            // Divide each feature by total length
            for (Integer feature : example.getFeatureSet()) {
                Double oldFeatureValue = example.getFeature(feature);
                example.setFeature(feature, oldFeatureValue/totalLength);
            }
        }
    }

    /**
     * Preprocess the test dataset by going through each example and normalizing all the features
     * @param test, the test dataset to normalize
     */
    @Override
    public void preprocessTest(DataSet test) {
        // Go through all the examples in the data set

        // For each example, normalize

        for (Example example : test.getData()) {
            // Record total length
            Double totalLength = 0.0;
            for (Integer feature : example.getFeatureSet()) {
                // Get the length
                totalLength += Math.pow(example.getFeature(feature), 2);
            }
            // Square root the length
            totalLength = Math.pow(totalLength, 0.5);
            // Divide each feature by total length
            for (Integer feature : example.getFeatureSet()) {
                Double oldFeatureValue = example.getFeature(feature);
                example.setFeature(feature, oldFeatureValue/totalLength);
            }
        }
    }
    
}
