import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.Debug.Random;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemovePercentage;

public class Main {

	public static void main(String[] args) throws Exception {
		String file="lib/iris.csv";
		DataSource source = new DataSource(file);
		Instances data = source.getDataSet();
		
		// Set up cross validation data set for use
		// This data is the same as above, but we use the whole set multiple times.
		Instances cross_validation_data = data;
		cross_validation_data.setClassIndex(cross_validation_data.numAttributes() - 1);
		
		data.randomize(new Random());
		
		RemovePercentage filter = new RemovePercentage();
		filter.setPercentage(30);
		filter.setInputFormat(data);
		Instances training = Filter.useFilter(data, filter);
		
		System.err.println("count of training: " + training.numInstances());
		
		// Make the last attribute be the class
		training.setClassIndex(training.numAttributes() - 1);
		filter.setInputFormat(data);
		filter.setInvertSelection(true);
		Instances test = Filter.useFilter(data, filter);
		
		System.err.println("count of test: " + test.numInstances());
		
		// Make the last attribute be the class
		test.setClassIndex(test.numAttributes() - 1);
		
		HardCodedClassifier classifier = new HardCodedClassifier();
		
		Evaluation eval = new Evaluation(training);
		eval.evaluateModel(classifier, test);

		String summary = eval.toSummaryString();
		System.out.print(summary);
		
		// Attempt at running a cross validation check using my classifier
		Evaluation multi_eval = new Evaluation(cross_validation_data);
		multi_eval.crossValidateModel(classifier, cross_validation_data, 10, new Random());
		System.out.println(multi_eval.toSummaryString("\n\nResults\n\n", false));
	}
}
