package com.predictionmarketing.RecommenderApp;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map.Entry;
import java.util.Random;
import java.util.StringTokenizer;

import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.impl.recommender.svd.ALSWRFactorizer;
import org.apache.mahout.cf.taste.impl.recommender.svd.Factorization;
import org.apache.mahout.cf.taste.model.DataModel;

import com.predictionmarketing.RecommenderApp.Misc.MyEntry;
import com.predictionmarketing.RecommenderApp.Misc.MyPair;
import com.predictionmarketing.RecommenderApp.Misc.RMSECalculator;
import com.predictionmarketing.RecommenderApp.RecSystems.GlobalAverage;
import com.predictionmarketing.RecommenderApp.RecSystems.PerArtistAverage;
import com.predictionmarketing.RecommenderApp.RecSystems.RecSystemInterface;
import com.predictionmarketing.RecommenderApp.RecSystems.SimpleFactorization;

public class ALSTest {
	
	// number of folds for cross-validation
	static int K = 10;
	static int NUMBER_OF_ITERATIONS = 10;
	static double lambdas[] = new double[]{750};
	static int numOfFeatures = 20;
	
	public static void main(String[] args) throws Exception {
		int i,j,k;
		
		// we read all the data
		DataModel model = new FileDataModel(new File("data/pairs.dat"));
		
		// we place the relationships in a hashmap
		HashMap<String, Long> hmRelationships = new HashMap<String, Long>();
		BufferedReader br = new BufferedReader(new FileReader("data/pairs.dat"));
		while (true) {
			String line = br.readLine();
			if (line == null) {
				break;
			}
			StringTokenizer st = new StringTokenizer(line, ",");
			hmRelationships.put(st.nextToken()+" "+st.nextToken(), parseLong(st.nextToken()));
		}
		br.close();
		
		// we put all the relationships in array
		MyPair pairs[] = new MyPair[hmRelationships.size()];
		br = new BufferedReader(new FileReader("data/pairs.dat"));
		i = 0;
		while (true) {
			String line = br.readLine();
			if (line == null) {
				break;
			}
			StringTokenizer st = new StringTokenizer(line, ",");
			pairs[i] = new MyPair(Long.parseLong(st.nextToken()), Long.parseLong(st.nextToken()), parseLong(st.nextToken()));
			i++;
		}
		br.close();
		
		// we put all the relationships in array list
		ArrayList<MyPair> alPairs = new ArrayList<MyPair>();
		for (i=0;i<pairs.length;i++) {
			alPairs.add(pairs[i]);
		}
		
		Random r = new Random(System.currentTimeMillis());
		alPairs = randomize(alPairs, r);
		
		// first average predictions
		// Question: This is average, not random, right?
		double rmse1 = 0;
		RecSystemInterface rec = new GlobalAverage();
		RMSECalculator calc = new RMSECalculator();
		for (k=1;k<=K;k++) {
			ArrayList<MyPair> trainingPairs = getPart(alPairs, r, k, K, false);
			ArrayList<MyPair> testPairs = getPart(alPairs, r, k, K, true);
			rec.Train(trainingPairs);
			double rmse = calc.computeRMSE(testPairs, rec.Predict(testPairs));
			System.out.println(rmse);
			rmse1 += (rmse / K);
		}
		System.out.println("RMSE (average 1): "+rmse1);
		
		
		// second - average per artist prediction
		double rmse2 = 0;
		rec = new PerArtistAverage();
		calc = new RMSECalculator();
		for (k=1;k<=K;k++) {
			ArrayList<MyPair> trainingPairs = getPart(alPairs, r, k, K, false);
			ArrayList<MyPair> testPairs = getPart(alPairs, r, k, K, true);
			rec.Train(trainingPairs);
			double rmse = calc.computeRMSE(testPairs, rec.Predict(testPairs));
			System.out.println(rmse);
			rmse2 += (rmse / K);
		}
		System.out.println("RMSE (average per artist 2): "+rmse2);
		
				
		double opts[] = new double[lambdas.length];
		
		// let's use cross-validation to find the optimal lambda
		for (i=0;i<lambdas.length;i++) {
			// for each different value of lambda we use CV
			// currently we are working with lambdas[i]
			
			double avg = 0;
			
			rec = new SimpleFactorization();
			HashMap<String, Object> parameters = new HashMap<String, Object>();
			parameters.put("numOfFeatures", numOfFeatures);
			parameters.put("numOfIterations", NUMBER_OF_ITERATIONS);
			parameters.put("lambda", lambdas[i]);
			rec.setParameters(parameters);
			
			for (k=1;k<=K;k++) {
				System.out.println("Lambda = " + lambdas[i] + " Fold = " + k);
				ArrayList<MyPair> trainingPairs = getPart(alPairs, r, k, K, false);
				ArrayList<MyPair> testPairs = getPart(alPairs, r, k, K, true);
				rec.Train(trainingPairs);
				double rmse = calc.computeRMSE(testPairs, rec.Predict(testPairs));
				System.out.println(rmse);
				System.out.println(calc.getPredictedPercentage() * 100 + "%");
				avg += (rmse / K);
			}
			opts[i] = avg;
		}
		
		for (i=0;i<lambdas.length;i++) {
			System.out.println(lambdas[i]+"\t"+opts[i]);
		}
		
	}
	

	static String getStringMapping(long userID, long itemID) {
		StringBuilder sb = new StringBuilder();
		sb.append(userID);
		sb.append(' ');
		sb.append(itemID);
		return sb.toString();
	}
	
	static MyEntry<Long, Long> getOriginal(String s) {
		StringTokenizer st = new StringTokenizer(s);
		MyEntry<Long, Long> entry = new MyEntry<Long, Long>(Long.parseLong(st.nextToken()), Long.parseLong(st.nextToken()));
		return entry;
	}
	
	static long parseLong(String s) {
		long num = 0;
		if (s.contains("e")) {
			double numD = Double.parseDouble(s.substring(0, s.indexOf('e')));
			int degree = Integer.parseInt(s.substring(s.indexOf('+')+1));
			for (int j = 0;j<degree;j++) {
				numD *= 10;
			}
			num = (long)numD;
		} else {
			num = Long.parseLong(s);
		}
		return num;
	}
	
	static ArrayList<MyPair> randomize(ArrayList<MyPair> originalPairs, Random r) {
		int i,j,k;
		int N = originalPairs.size();
		
		ArrayList<MyPair> randomizedPairs = new ArrayList<MyPair>();
		
		while (!originalPairs.isEmpty()) {
			randomizedPairs.add(originalPairs.remove(r.nextInt(originalPairs.size())));
		}
		
		return randomizedPairs;
	}
	
	// we select k-th of total number of equal parts, ex. 1 of 4, 2 of 4, 3 of 4, 4 of 4
	public static ArrayList<MyPair> getPart(ArrayList<MyPair> originalPairs, Random r, int k, int total, boolean selected) {
		int i,j;
		int N = originalPairs.size();
		
		ArrayList<MyPair> selectedPairs = new ArrayList<MyPair>();
		int partSize = N / total;
		int modulo = N % total;
		
		Iterator<MyPair> it = originalPairs.iterator();
		
		int soFar = 0;
		int currentPart = 1;
		
		for (i=0;i<N;i++) {
			MyPair current = it.next();
			soFar++;
			
			//System.out.println(currentPart+" "+soFar);
			
			if (selected == true) {
				if (currentPart == k) {
					selectedPairs.add(current);
				}
			} else {
				if (currentPart != k) {
					selectedPairs.add(current);
				}
			}
			
			if (currentPart <= modulo) {
				if (soFar == partSize+1) {
					currentPart++;
					soFar = 0;
				}
			} else {
				if (soFar == partSize) {
					currentPart++;
					soFar = 0;
				}
			}
			
		}
		
		return selectedPairs;
	}
}