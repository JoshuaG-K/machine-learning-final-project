import ml.classifiers.TwoLayerNN;
import ml.data.DataSet;
import ml.data.DataSetSplit;
import ml.data.FeatureNormalizer;

/**
 * Class that implements the PCA algorithm with SVD and eigen vector covariance matrix
 * 
 * @author Joshua Garcia-Kimble, Abraham Arias, Ash Shah
 */
public class PCA {

    public static void main(String[] args) {
        String dataFolderPath = "C:\\Users\\Joshua\\Documents\\College\\Senior\\MachineLearning\\final-project\\machine-learning-final-project\\data\\";
        String dataTrainFileName = "cs-training.csv";
        String dataTrainFilePath = dataFolderPath + dataTrainFileName;

        DataSet dataTrain = new DataSet(dataTrainFilePath, DataSet.CSVFILE);
        DataSetSplit dataSplit = dataTrain.split(0.1);
        DataSet dataTrainSmall = dataSplit.getTrain();
        PCA pca = new PCA(dataTrainFilePath, 5, PCA_Type.EIGEN);
        pca.getAccuracyOnFullData(dataTrainSmall);
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

    private void Eigen_Pca() {
        // TODO: Calculate the covariance matrix

        // 
    }

    private void Svd_Pca() {

    }


}