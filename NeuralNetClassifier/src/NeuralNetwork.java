import java.util.ArrayList;
import java.util.Collections;

import weka.core.Instance;


public class NeuralNetwork {
	
	ArrayList<ArrayList<Neuron>> layers;
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
						layers.get(j).add(new Neuron(true));
					}
				}
			}
		}
	}
	
	public double evaluate (Instance inst) {
		int count = inst.numAttributes();
		
		ArrayList<Double> inputs = new ArrayList<Double>();
		ArrayList<ArrayList<Double>> activations = new ArrayList<ArrayList<Double>>();
		
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
}
