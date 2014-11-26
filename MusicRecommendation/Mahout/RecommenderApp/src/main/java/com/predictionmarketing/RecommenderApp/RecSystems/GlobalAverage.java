package com.predictionmarketing.RecommenderApp.RecSystems;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.Map;

import com.predictionmarketing.RecommenderApp.Misc.MyPair;

public class GlobalAverage implements RecSystemInterface {

	private double avg;
	
	public void Train(ArrayList<MyPair> trainingPairs) {
		
		// Model training
		Iterator<MyPair> it = trainingPairs.iterator();
		avg = 0;
		while (it.hasNext()) {
			avg += it.next().value;
		}
		avg /= trainingPairs.size();
	}
	
	public ArrayList<MyPair> Predict(ArrayList<MyPair> testPairs) {
		// Test error estimation
		ArrayList<MyPair> results = new ArrayList<MyPair>();
		Iterator<MyPair> it = testPairs.iterator();
		while (it.hasNext()) {
			MyPair pair = new MyPair(it.next());
			pair.value = avg;
			results.add(pair);
		}
		return results;
	}

	public void setParameters(Map<String, Object> parameters) {
	}

	public void printModel() {
		System.out.println("Avg = " + avg);
	}
}
