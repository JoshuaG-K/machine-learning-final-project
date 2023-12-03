public class TopKEigenValuesAndVectors {
    private int k;
    private double[] eigenValues;
    private double[][] eigenVectors;

    public TopKEigenValuesAndVectors(int k, double[] eigenValues, double[][] eigenVectors) {
        this.k = k;
        this.eigenValues = eigenValues;
        this.eigenVectors = eigenVectors;
    }

    public int getK() {
        return k;
    }

    public double[] getEigenValues() {
        return eigenValues;
    }

    public double[][] getEigenVectors() {
        return eigenVectors;
    }
}
