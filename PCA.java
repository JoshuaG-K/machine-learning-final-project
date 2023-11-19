import ml.data.DataSet;
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
    }

    /** 
     * Used to specify the type of PCA we want to do.
     * We support the SVD and Eigen Vectors of the Covariance Matrix methods.
     */
    public enum PCA_Type {
        SVD,
        EIGEN
    }

    // The type of PCA to do
    private PCA_Type pcaType;

    // The k most vectors we want from the algorithm
    private int k;

    // The data we will do pca on
    private DataSet data;

    /**
     * The constructor assumes the data is in a csv file and that we want the k most important feature combinations
     */
    public PCA(DataSet data, int k, PCA_Type pcaType) {
        FeatureNormalizer featureNormalizer = new FeatureNormalizer();
        this.data = data;
        // We normalize the features since this will make our Eigen vector pca calculations better 
        featureNormalizer.preprocessTrain(this.data);;
        this.k = k;
        this.pcaType = pcaType;

        if (this.pcaType == PCA_Type.EIGEN) {

        } else if (this.pcaType == PCA_Type.SVD) {

        }

        /**
         * TODO: Run PCA in the constructor. I think we should make one function for SVD PCA and one function for Eigen covariance matrix PCA 
         * then just call those from here. 
         */
        
    }

    private void Eigen_Pca() {
        // TODO: Calculate the covariance matrix

        // 
    }

    private void Svd_Pca() {

    }


}