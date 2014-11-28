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
import org.apache.mahout.cf.taste.impl.neighborhood.NearestNUserNeighborhood;
import org.apache.mahout.cf.taste.impl.neighborhood.ThresholdUserNeighborhood;
import org.apache.mahout.cf.taste.impl.recommender.GenericUserBasedRecommender;
import org.apache.mahout.cf.taste.impl.similarity.PearsonCorrelationSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.neighborhood.UserNeighborhood;
import org.apache.mahout.cf.taste.recommender.UserBasedRecommender;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;

import com.predictionmarketing.RecommenderApp.Misc.MyPair;

public class SimpleNeighborhood implements RecSystemInterface {

	private int N;
	private UserBasedRecommender recommender;
	
	public void Train(ArrayList<MyPair> trainingPairs) {
		 DataModel model = null;
		 try {
			model = getDataModel(trainingPairs);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	     UserSimilarity similarity = null;
	     try {
			similarity = new PearsonCorrelationSimilarity(model);
		} catch (TasteException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	     UserNeighborhood neighborhood = null;
	     try {
			neighborhood = new NearestNUserNeighborhood(N, similarity, model);
		} catch (TasteException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	     recommender = new GenericUserBasedRecommender(model, neighborhood, similarity);
	}

	public ArrayList<MyPair> Predict(ArrayList<MyPair> testPairs) {
		ArrayList<MyPair> ret = new ArrayList<MyPair>();
		for(MyPair pair : testPairs) {
			MyPair put = new MyPair(pair);
			try {
				put.value = recommender.estimatePreference(pair.userID, pair.itemID);
				if (!Double.isNaN(put.value)) {
					ret.add(put);
				}
			} catch (TasteException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		return ret;
	}

	public void setParameters(Map<String, Object> parameters) {
		N = Integer.parseInt((String)parameters.get("N"));

	}

	public void printModel() {
	}
	
	static DataModel getDataModel(ArrayList<MyPair> pairs) throws Exception {			
		PrintWriter out = new PrintWriter(new File("dummy.dat"));
		Iterator<MyPair> it = pairs.iterator();
		while (it.hasNext()) {
			MyPair current = it.next();
			out.println(current.userID+","+current.itemID+","+current.value);
		}
		out.close();
		return new FileDataModel(new File("dummy.dat"));
	}

}
