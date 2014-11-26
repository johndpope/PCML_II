package com.predictionmarketing.RecommenderApp.RecSystems;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

import com.predictionmarketing.RecommenderApp.Misc.MyPair;

public class PerArtistAverage implements RecSystemInterface {

	private HashMap<Long, Double> sumPerArtist;
	private HashMap<Long, Long> countPerArtist;
	
	public void Train(ArrayList<MyPair> trainingPairs) {
		sumPerArtist = new HashMap<Long, Double>();
		countPerArtist = new HashMap<Long, Long>();
		
		// Model training
		Iterator<MyPair> it = trainingPairs.iterator();
		while (it.hasNext()) {
			MyPair pair = it.next();
			if (!sumPerArtist.containsKey(pair.itemID)) {
				sumPerArtist.put(pair.itemID, 0.0);
				countPerArtist.put(pair.itemID, 0L);
			}
			sumPerArtist.put(pair.itemID, sumPerArtist.get(pair.itemID) + pair.value);
			countPerArtist.put(pair.itemID, countPerArtist.get(pair.itemID) + 1);
		}
	}
	
	public void setParameters(Map<String, Object> parameters) {
	}


	public ArrayList<MyPair> Predict(ArrayList<MyPair> testPairs) {
		// Test RMSE
		Iterator<MyPair> it = testPairs.iterator();
		ArrayList<MyPair> results = new ArrayList<MyPair>();
		while (it.hasNext()) {
			MyPair pair = new MyPair(it.next());
			if (sumPerArtist.containsKey(pair.itemID)) {
				double predictedValue = sumPerArtist.get(pair.itemID) / countPerArtist.get(pair.itemID);
				pair.value = predictedValue;
				results.add(pair);
			}
		}
		return results;
	}

	public void printModel() {
	}

}
