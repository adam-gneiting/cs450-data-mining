import java.util.ArrayList;
import java.util.Arrays;

import weka.core.Instances;
import weka.core.Debug.Random;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemovePercentage;


public class Main {
	
	private static int runs = 1000;
	private static ArrayList<Integer> layers = new ArrayList<Integer>(Arrays.asList(5,4)); 
	
	public static void main(String[] args) throws Exception {
		String file="lib/iris.csv";
		DataSource source = new DataSource(file);
		Instances data = source.getDataSet();
		
		data.randomize(new Random());
		
		RemovePercentage filter = new RemovePercentage();
		filter.setPercentage(30);
		filter.setInputFormat(data);
		Instances training = Filter.useFilter(data, filter);
		int training_size = training.numInstances();
		
		System.err.println("count of training: " + training.numInstances());
		
		// Make the last attribute be the class
		training.setClassIndex(training.numAttributes() - 1);
		filter.setInputFormat(data);
		filter.setInvertSelection(true);
		Instances test = Filter.useFilter(data, filter);
		int test_size = test.numInstances();
		
		System.err.println("count of test: " + test.numInstances());
		
		// Make the last attribute be the class
		test.setClassIndex(test.numAttributes() - 1);
		
		NeuralNetwork nn = new NeuralNetwork(layers);
		
		for (int i = 0; i < runs; ++i) {
			for (int j = 0; j < training_size; ++j) {
				nn.evaluate(training.instance(j));
				nn.backPropagate();
			}
		}
		
		for (int i = 0; i < test_size; ++i) {
			Double class_index = nn.evaluate(test.instance(i));
			boolean success = nn.classificationValidation(class_index);
			System.out.println(((success) ? "Success" : "Failure"));
		}
	}
}
