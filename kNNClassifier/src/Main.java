import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.Debug.Random;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Standardize;
import weka.filters.unsupervised.instance.RemovePercentage;


public class Main {

	public static void main(String[] args) throws Exception {
		System.out.println("Classifying iris.csv");
		classifyFile("lib/iris.csv");
		System.out.println("Classifying car.csv");
		classifyFile("lib/car.csv");
		System.out.println("Classifying breast-cancer-wisconsin.csv");
		classifyFile("lib/breast-cancer-wisconsin.csv");
	}
	
	private static void classifyFile(String file) throws Exception {
		DataSource source = new DataSource(file);
		Instances data = source.getDataSet();
		
		Standardize standardizeData = new Standardize();
		standardizeData.setInputFormat(data);
		
		data.randomize(new Random());
		
		// Set up cross validation data set for use
		// This data is the same as above, but we use the whole set multiple times.
		Instances cross_validation_data = data;
		cross_validation_data.setClassIndex(cross_validation_data.numAttributes() - 1);
		
		RemovePercentage filter = new RemovePercentage();
		
		filter.setPercentage(30);
		filter.setInputFormat(data);
		
		Instances training = Filter.useFilter(data, filter);
		// Make the last attribute be the class
		training.setClassIndex(training.numAttributes() - 1);
		
		filter.setInputFormat(data);
		filter.setInvertSelection(true);
		
		Instances test = Filter.useFilter(data, filter);
		// Make the last attribute be the class
		test.setClassIndex(test.numAttributes() - 1);
		
		KNNClassifier classifier = new KNNClassifier();
		classifier.buildClassifier(training);
		
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
