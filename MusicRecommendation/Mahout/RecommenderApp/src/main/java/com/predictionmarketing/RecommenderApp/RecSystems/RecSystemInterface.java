package com.predictionmarketing.RecommenderApp.RecSystems;

import java.util.ArrayList;
import java.util.Map;

import com.predictionmarketing.RecommenderApp.Misc.MyPair;

public interface RecSystemInterface {
	void Train(ArrayList<MyPair> trainingPairs);
	ArrayList<MyPair> Predict(ArrayList<MyPair> testPairs);
	void setParameters(Map<String, Object> parameters);
	void printModel();
}
