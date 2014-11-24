package com.predictionmarketing.RecommenderApp;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;
import java.util.StringTokenizer;

import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.impl.recommender.svd.ALSWRFactorizer;
import org.apache.mahout.cf.taste.impl.recommender.svd.Factorization;
import org.apache.mahout.cf.taste.model.DataModel;

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
		
		// first random predictions
		double rmse1 = 0;
		for (k=1;k<=K;k++) {
			ArrayList<MyPair> trainingPairs = getPart(alPairs, r, k, K, false);
			ArrayList<MyPair> testPairs = getPart(alPairs, r, k, K, true);
			double rmse = trainAndPredictRandom1(trainingPairs, testPairs);
			System.out.println(rmse);
			rmse1 += (rmse / K);
		}
		System.out.println("RMSE (random 1): "+rmse1);
		
		double opts[] = new double[lambdas.length];
		
		// let's use cross-validation to find the optimal lambda
		for (i=0;i<lambdas.length;i++) {
			// for each different value of lambda we use CV
			// currently we are working with lambdas[i]
			
			double avg = 0;
			
			for (k=1;k<=K;k++) {
				ArrayList<MyPair> trainingPairs = getPart(alPairs, r, k, K, false);
				ArrayList<MyPair> testPairs = getPart(alPairs, r, k, K, true);
				double rmse = trainAndPredict(trainingPairs, testPairs, numOfFeatures, lambdas[i], NUMBER_OF_ITERATIONS);
				System.out.println(rmse);
				avg += (rmse / K);
			}
			opts[i] = avg;
		}
		
		for (i=0;i<lambdas.length;i++) {
			System.out.println(lambdas[i]+"\t"+opts[i]);
		}
		
	}
	
	static double trainAndPredict(ArrayList<MyPair> trainingPairs, ArrayList<MyPair> testPairs, int numOfFeatures, double lambda, int numOfIterations) throws Exception {

		DataModel trainingData = getDataModel(trainingPairs);
		System.out.println("SUCCESS");
		
		long t1 = System.currentTimeMillis();
		ALSWRFactorizer fact = new ALSWRFactorizer(trainingData, numOfFeatures, lambda, numOfIterations);
		
		Factorization factorization = fact.factorize();
		
		double itemFeatures[][] = factorization.allItemFeatures();
		double userFeatures[][] = factorization.allUserFeatures();
		
		HashMap<Long, Integer> hmItemsOriginalToMapped = new HashMap<Long, Integer>();
		HashMap<Integer, Long> hmItemsMappedToOriginal = new HashMap<Integer, Long>();
		
		Iterable<Entry<Long, Integer>> iter = factorization.getItemIDMappings();
		Iterator<Entry<Long, Integer>> it = iter.iterator();
		
		while (it.hasNext()) {
			Entry<Long, Integer> current = it.next();
			hmItemsOriginalToMapped.put(current.getKey(), current.getValue());
			hmItemsMappedToOriginal.put(current.getValue(), current.getKey());
		}
		
		HashMap<Long, Integer> hmUsersOriginalToMapped = new HashMap<Long, Integer>();
		HashMap<Integer, Long> hmUsersMappedToOriginal = new HashMap<Integer, Long>();
		
		iter = factorization.getUserIDMappings();
		it = iter.iterator();
		
		while (it.hasNext()) {
			Entry<Long, Integer> current = it.next();
			hmUsersOriginalToMapped.put(current.getKey(), current.getValue());
			hmUsersMappedToOriginal.put(current.getValue(), current.getKey());
		}
		
		long t2 = System.currentTimeMillis();
		long totalTime = t2 - t1;
		
		System.out.println("The program completed in "+totalTime+" milliseconds.");
		
		double rmse = calculateRMSE(userFeatures, itemFeatures, testPairs, hmUsersOriginalToMapped, hmItemsOriginalToMapped);
		return rmse;
	}
	
	static double calculateRMSE(double userFeatures[][], double itemFeatures[][], ArrayList<MyPair> testPairs, HashMap<Long, Integer> hmUsersOriginalToMapped, HashMap<Long, Integer> hmItemsOriginalToMapped) {
		double score = 0;
		int i;
		int N = userFeatures.length;
		int M = itemFeatures.length;
		
		int included = 0;
		
		Iterator<MyPair> it = testPairs.iterator();
		while (it.hasNext()) {
			MyPair current = it.next();
			if ((hmUsersOriginalToMapped.get(current.userID) != null)&&(hmItemsOriginalToMapped.get(current.itemID) != null)) {
				double trueValue = current.value;
				double predictedValue = (long)getProduct(userFeatures, itemFeatures,
					hmUsersOriginalToMapped.get(current.userID), hmItemsOriginalToMapped.get(current.itemID));
				//System.out.println(trueValue+" "+predictedValue);
				score += (trueValue - predictedValue) * (trueValue - predictedValue);
				included++;
			}
		}
		score = Math.sqrt(score);
		System.out.println(included+"/"+testPairs.size());
		
		return score;
	}
	
	static double trainAndPredictRandom1(ArrayList<MyPair> trainingPairs, ArrayList<MyPair> testPairs) {
		
		Iterator<MyPair> it = trainingPairs.iterator();
		double avg = 0;
		while (it.hasNext()) {
			avg += it.next().value;
		}
		avg /= trainingPairs.size();
		
		double rmse = 0;
		it = testPairs.iterator();
		while (it.hasNext()) {
			double current = it.next().value;
			rmse += (current - avg) * (current - avg);
		}
		rmse = Math.sqrt(rmse);
		
		return rmse;
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
	
}

class MyEntry<K, V> implements Map.Entry<K, V> {
    private final K key;
    private V value;

    public MyEntry(K key, V value) {
        this.key = key;
        this.value = value;
    }

    public K getKey() {
        return key;
    }

    public V getValue() {
        return value;
    }

    public V setValue(V value) {
        V old = this.value;
        this.value = value;
        return old;
    }
}

class MyPair {
	long userID;
	long itemID;
	long value;
	MyPair(long userID, long itemID, long value) {
		this.userID = userID;
		this.itemID = itemID;
		this.value = value;
	}
	MyPair(MyPair p) {
		userID = p.userID;
		itemID = p.itemID;
		value = p.value;
	}
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(userID);
		sb.append(", ");
		sb.append(itemID);
		sb.append(", ");
		sb.append(value);
		return sb.toString();
	}
}