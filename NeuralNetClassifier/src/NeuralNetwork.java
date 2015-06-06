import java.util.ArrayList;
import java.util.Collections;

import weka.core.Instance;


public class NeuralNetwork {
	
	ArrayList<ArrayList<Neuron>> layers;
	ArrayList<ArrayList<Double>> activations;
	// TODO: This forces sequential propagation.  Update to allow batch
	// processing.
	Instance instance;
	int length;
	
	public NeuralNetwork(ArrayList<Integer> layer_counts) {
		length = layer_counts.size();
		Boolean bias = (length > 1);
				
		for (int i = 0; i < length; ++i) {
			int count = layer_counts.get(i);
			boolean bias_added = !bias;
			
			if (bias) {
				count++;
			}
			
			layers.add(new ArrayList<Neuron>());
			
			for (int j = 0; j < count; ++j) {
				if (bias_added) {
					layers.get(j).add(new Neuron());
				} else {
					bias_added = !bias_added;
					if ((i + 1) == length) {
						Neuron temp = new Neuron(true);
						layers.get(j).add(temp);
					}
				}
			}
		}
	}
	
	public double evaluate (Instance inst) {
		instance = inst;
		int count = inst.numAttributes();
		
		ArrayList<Double> inputs = new ArrayList<Double>();
		activations = new ArrayList<ArrayList<Double>>();
		
		for (int i = 0; i < count; ++i) {
			inputs.add(inst.value(i));
		}
		activations.add(inputs);
		
		for (int i = 0; i < length; ++i) {
			ArrayList<Double> layer_outputs = new ArrayList<Double>();
			for (Neuron n : layers.get(i)) {
				layer_outputs.add(n.evaluate(inputs));
			}
			inputs = layer_outputs;
			activations.add(layer_outputs);
		}
		
		Double max_value = Collections.max(inputs);
		return inputs.indexOf(max_value);
	}
	
	public void backPropagate() {
		boolean output_layer = true;
		ArrayList<ArrayList<Double>> errors = new ArrayList<ArrayList<Double>>();
		for (int i = layers.size(); i > 0; --i) {
			ArrayList<Neuron> layer = layers.get(i - 1);
			ArrayList<Double> layer_errors = new ArrayList<Double>();
			ArrayList<Double> new_layer_errors = new ArrayList<Double>();
			ArrayList<Double> inputs = activations.get(i - 1);
			int size = layer.size();
			if (!output_layer) {
				// Populate errors list
				layer_errors = errors.get(size - i);
			}
			
			for (int j = 0; j < size; ++j) {
				Neuron n = layer.get(j);
				ArrayList<Neuron> next_layer = layers.get(i);
				boolean correct_class = ((int) instance.classValue() == j);
				
//				ArrayList<Double> weighted_errors = new ArrayList<Double>();
//				int next_size = next_layer.size();
//				for (int k = 0; k < next_size; ++k) {
//					Neuron prev = next_layer.get(k);
//					weighted_errors.add(prev.getError() * prev.getWeightByIndex(j));
//				}
				// Get the unweighted errors for the Neuron.
				Double error = n.getError(output_layer, j, next_layer, correct_class);
				
				new_layer_errors.add(error);
				n.updateWeights(inputs);
			}
			
			// Save off our new errors for use in next iteration.
			errors.add(new_layer_errors);
			
			// Once the Last layer (first execution) has been run
			// we need to perform a different update function.
			output_layer = false;
		}
	}
}
