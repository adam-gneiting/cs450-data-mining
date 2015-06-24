import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
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
		
		TreeMap<Double, List<Instance>> map = new TreeMap<Double, List<Instance>>();
		for (int j = 0; j < this.mInstances.numInstances(); ++j) {
			Instance compared = this.mInstances.instance(j);
			Double distance = this.getDistance(i, compared);
			List<Instance> list = new ArrayList<Instance>();
			if (map.get(distance) != null) {
				 list = map.get(distance);
			}
			list.add(compared);
			map.put(distance, (ArrayList<Instance>) list);
		}
		
		List<Double> classes = new ArrayList<Double>();
		for (int j = 0; j < this.k; ++j) {
			List<Instance> tmp_list = (List<Instance>) map.pollFirstEntry().getValue();
			j += tmp_list.size() - 1;

			for (Instance tmp_inst : tmp_list) {
				classes.add(tmp_inst.classValue());
			}
		}
		
		return this.mostCommon(classes);
	}
	
	private double getDistance(Instance right, Instance left) {
		double distance = 0.0;
		
		for(int i = 1; i < right.numAttributes(); ++i) {
			// numeric values fit as Manhattan distance
			if (right.attribute(i).isNumeric()) {
				distance += Math.abs(right.value(i) - left.value(i));
			} else {
				distance += (right.value(i) == left.value(i)) ? this.weightOfNominal : 0;
			}
		}
		
		return distance;
	}
	
	private <T> T mostCommon(List<T> list) {
	    Map<T, Integer> map = new HashMap<>();

	    for (T t : list) {
	        Integer val = map.get(t);
	        map.put(t, val == null ? 1 : val + 1);
	    }

	    Entry<T, Integer> max = null;

	    for (Entry<T, Integer> e : map.entrySet()) {
	        if (max == null || e.getValue() > max.getValue())
	            max = e;
	    }

	    return max.getKey();
	}
}