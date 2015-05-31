import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.TreeMap;
import java.lang.Math;

@SuppressWarnings("serial")
public class KNNClassifier extends Classifier {
	
	private Instances mInstances;
	private double weightOfNominal;
	private int k = 5;
	
	@Override
	public void buildClassifier(Instances pInstances) throws Exception {
		this.mInstances = pInstances;
		this.weightOfNominal = 1;
	}
	
	public void buildClassifier(Instances pInstances, double nominalWeight) throws Exception{
		this.buildClassifier(pInstances);
		this.weightOfNominal = nominalWeight;
	}
	
	@Override
	public double classifyInstance(Instance i) {
		
		TreeMap<Double, Instance> map = new TreeMap<Double, Instance>();
		for (int j = 0; j < this.mInstances.numInstances(); ++j) {
			Instance compared = this.mInstances.instance(j);
			Double distance = this.getDistance(i, compared);
			map.put(distance, compared);
		}
		
		TreeMap<Double, ArrayList<Instance>> classes = new TreeMap<Double, ArrayList<Instance>>();
		for (int j = 0; j < this.k; ++j) {
			Instance tmp = (Instance) map.pollFirstEntry().getValue();
			ArrayList<Instance> list = new ArrayList<Instance>();
			Double classValue = tmp.classValue();
			if (classes.get(classValue) != null) {
				 list = classes.get(tmp.classValue());
			}
			list.add(tmp);
			classes.put(tmp.classValue(), list);
		}
		
		ArrayList<Instance> instances = classes.pollLastEntry().getValue();
		return instances.remove(0).classValue();
	}
	
	private double getDistance(Instance right, Instance left) {
		double distance = 0.0;
		
		for(int i = 1; i < right.numAttributes(); ++i) {
			// numeric values fit as Manhattan distance
			if (right.attribute(i).isNumeric()) {
				distance += Math.abs(right.value(i) - left.value(i));
			} else {
				distance += (right.attribute(i).equals(left.attribute(i))) ? this.weightOfNominal : 0;
			}
		}
		
		return distance;
	}
	
}