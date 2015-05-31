import java.util.ArrayList;


public class Neuron {

	ArrayList<Double> weights;
	boolean bias_neuron;
	
	public Neuron() {
		this.init(false);
	}
	
	public Neuron(boolean bias) {
		this.init(bias);
	}
	
	public void init(boolean bias) {
		bias_neuron = bias;
		weights = new ArrayList<Double>();
	}
	
	public Double evaluate(ArrayList<Double> inputs) {
		Double result = 0.0;
		if (!bias_neuron) {
			
			int length = inputs.size();
			
			if (this.weights.size() == 0) {
				this.init(length);
			}
			
			for (int i = 0; i < length; ++i) {
				result += weights.get(i) * inputs.get(i);
			}
			
			result = 1 / (1 + Math.pow(Math.E, -result));
		} else {
			// Bias Neurons always have an activation value of -1
			result = 1.0;
		}
		
		return result;
	}
	
	private void init(int length) {
		for (int i = 0; i < length; ++i) {
			weights.add(((Math.random() * 2) - 1) / length);
		}
	}
	
}
