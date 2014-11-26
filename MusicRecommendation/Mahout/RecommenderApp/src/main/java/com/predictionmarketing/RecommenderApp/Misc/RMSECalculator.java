package com.predictionmarketing.RecommenderApp.Misc;

import java.util.ArrayList;
import java.util.HashMap;

public class RMSECalculator {
	
	private HashMap<String, Double> truth;
	private int predictedCount;
	private int totalCount;
	private double rmse;
	
	public double computeRMSE(ArrayList<MyPair> truePairs,
							ArrayList<MyPair> predictedPairs) {
		predictedCount = 0;
		truth = new HashMap<String, Double>();
		for(MyPair pair : truePairs) {
			truth.put(pair.userID + " " + pair.itemID, pair.value);
		}
		totalCount = truePairs.size();
		rmse = 0;
		for(MyPair pair : predictedPairs) {
			String key = pair.userID + " " + pair.itemID;
			if (truth.containsKey(key)) {
				predictedCount++;
				rmse += (truth.get(key) - pair.value) * (truth.get(key) - pair.value); 
			}
		}
		rmse = Math.sqrt(rmse / predictedCount);
		return rmse;
	}
	
	public double getRMSE() { return rmse; }
	public double getPredictedPercentage() { return 1.0 * predictedCount / totalCount; }
}
