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
	}
}
