import java.util.ArrayList;


public class Neuron {

	static Double n_const = 0.9;
	ArrayList<Double> weights = new ArrayList<Double>();
	Double activation_value;
	Double h_value;
	Double error;
	boolean bias_neuron;
	
	public Neuron() {
		this.init(false);
	}
	
	public Neuron(boolean bias) {
		this.init(bias);
	}
	
	public void init(boolean bias) {
		bias_neuron = bias;
	}
	
	public Double evaluate(ArrayList<Double> inputs) {
		Double result = 0.0;
		if (!bias_neuron) {
			
			int length = inputs.size();
			
			if (this.weights.size() == 0) {
				//System.err.println("Creating Weights array or length " + length);
				this.initWeights(length);
			}
			
			for (int i = 0; i < length; ++i) {
				result += weights.get(i) * inputs.get(i);
			}
			
			h_value = result;
			result = 1 / (1 + Math.pow(Math.E, -result));
		} else {
			// Bias Neurons always have an activation value of -1
			result = -1.0;
		}
		
		activation_value = result;
		//System.err.println("Result: " + result);
		
		return result;
	}
	
	public Double getWeightByIndex(int i) {
		return weights.get(i);
	}
	
	public Double getError() {
		return error;
	}
	
	public Double getError(boolean output_layer, int index, ArrayList<Neuron> next_layer, boolean correct_output) {
		
		if (output_layer) {
			this.outputLayerError(correct_output);
		} else {
			this.hiddenLayerError(index, next_layer);
		}
		
		return error;
	}
	
	public void updateWeights(ArrayList<Double> inputs) {
		if (!bias_neuron) {
			int size = inputs.size();
	
			//System.err.println("Size of 'weights': " + weights.size());
			for (int i = 0; i < size; ++i) {
				//System.err.println("Index: " + i);
				Double weight = weights.get(i);
				weights.set(i, weight - (n_const * error * inputs.get(i)));
			}
		}
	}
	
	public boolean is_bias() {
		return bias_neuron;
	}
	
	private void initWeights(int length) {
		for (int i = 0; i < length; ++i) {
			weights.add(((Math.random() * 2) - 1) / length);
		}
	}
	
	private void outputLayerError(boolean correct_output) {
		error = activation_value * (1 - activation_value) * (activation_value - ((correct_output) ? 1 : 0));
	}
	
	private void hiddenLayerError(int index, ArrayList<Neuron> next_layer) {
		if (!bias_neuron) {
		Double sum_of_weighted_errors = 0.0;
			int size = next_layer.size();
			for (int i = 0; i < size; ++i) {
				Neuron n = next_layer.get(i);
				if (!n.is_bias()) {
					Double weight = n.getWeightByIndex(index);
					Double error = n.getError();
					sum_of_weighted_errors += weight * error;
				}
			}
			
			error = activation_value * (1 - activation_value) * sum_of_weighted_errors;
		}
	}
}
