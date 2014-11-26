package com.predictionmarketing.RecommenderApp.RecSystems;

import java.io.File;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Map.Entry;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.impl.recommender.svd.ALSWRFactorizer;
import org.apache.mahout.cf.taste.impl.recommender.svd.Factorization;
import org.apache.mahout.cf.taste.model.DataModel;

import com.predictionmarketing.RecommenderApp.Misc.MyPair;

public class SimpleFactorization implements RecSystemInterface {

	private int numOfFeatures;
	private double lambda;
	private int numOfIterations;
	
	private double[][] itemFeatures;
	private double[][] userFeatures;
	
	private HashMap<Long, Integer> hmItemsOriginalToMapped;
	private HashMap<Integer, Long> hmItemsMappedToOriginal;
	
	private HashMap<Long, Integer> hmUsersOriginalToMapped;
	private HashMap<Integer, Long> hmUsersMappedToOriginal;
	
	static DataModel getDataModel(ArrayList<MyPair> pairs) throws Exception {			
			PrintWriter out = new PrintWriter(new File("data/dummy.dat"));
			Iterator<MyPair> it = pairs.iterator();
			while (it.hasNext()) {
				MyPair current = it.next();
				out.println(current.userID+","+current.itemID+","+current.value);
			}
			out.close();
			return new FileDataModel(new File("data/dummy.dat"));
		}
	
	
	public void Train(ArrayList<MyPair> trainingPairs) {
		DataModel trainingData = null;
		try {
			trainingData = getDataModel(trainingPairs);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		ALSWRFactorizer fact = null;
		try {
			fact = new ALSWRFactorizer(trainingData, numOfFeatures, lambda, numOfIterations);
		} catch (TasteException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		Factorization factorization = null;
		try {
			factorization = fact.factorize();
		} catch (TasteException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		itemFeatures = factorization.allItemFeatures();
		userFeatures = factorization.allUserFeatures();
		
		hmItemsOriginalToMapped = new HashMap<Long, Integer>();
		hmItemsMappedToOriginal = new HashMap<Integer, Long>();
		
		Iterable<Entry<Long, Integer>> iter = factorization.getItemIDMappings();
		Iterator<Entry<Long, Integer>> it = iter.iterator();
		
		while (it.hasNext()) {
			Entry<Long, Integer> current = it.next();
			hmItemsOriginalToMapped.put(current.getKey(), current.getValue());
			hmItemsMappedToOriginal.put(current.getValue(), current.getKey());
		}
		
		hmUsersOriginalToMapped = new HashMap<Long, Integer>();
		hmUsersMappedToOriginal = new HashMap<Integer, Long>();
		
		iter = factorization.getUserIDMappings();
		it = iter.iterator();
		
		while (it.hasNext()) {
			Entry<Long, Integer> current = it.next();
			hmUsersOriginalToMapped.put(current.getKey(), current.getValue());
			hmUsersMappedToOriginal.put(current.getValue(), current.getKey());
		}
	}
	
	static double getProduct(double userFeatures[][], double itemFeatures[][], int userIndex, int itemIndex) {
		int i,j,k;
		int numOfFeatures = userFeatures[0].length;
		
		double product = 0;
		for (i=0;i<numOfFeatures;i++) {
			product += userFeatures[userIndex][i]*itemFeatures[itemIndex][i];
		}
		
		return product;
	}
	

	public ArrayList<MyPair> Predict(ArrayList<MyPair> testPairs) {
		int i;
		int N = userFeatures.length;
		int M = itemFeatures.length;
		
		ArrayList<MyPair> ret = new ArrayList<MyPair>();
		
		Iterator<MyPair> it = testPairs.iterator();
		while (it.hasNext()) {
			MyPair current = new MyPair(it.next());
			if (hmUsersOriginalToMapped.containsKey(current.userID) && 
				hmItemsOriginalToMapped.containsKey(current.itemID)) {
				double predictedValue = (long)getProduct(userFeatures, itemFeatures,
					hmUsersOriginalToMapped.get(current.userID), hmItemsOriginalToMapped.get(current.itemID));
				current.value = predictedValue;
				ret.add(current);
			}
		}
		return ret;
	}

	public void setParameters(Map<String, Object> parameters) {
		numOfFeatures = Integer.parseInt((String)parameters.get("numOfFeatures"));
		lambda = Double.parseDouble((String)parameters.get("lambda"));
		numOfIterations = Integer.parseInt((String)parameters.get("numOfIterations"));
	}

	public void printModel() {
	}

}
