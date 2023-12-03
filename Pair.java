public class Pair implements Comparable<Pair>{
    private double value;
    private int index;

    public Pair(double value, int index) {
        this.value = value;
        this.index = index;
    }

    public double getValue() {
        return value;
    } 

    public int getIndex() {
        return index;
    }

    @Override
    public int compareTo(Pair other) {
        return Double.valueOf(value).compareTo(Double.valueOf(other.getValue()));
    }
    
}
